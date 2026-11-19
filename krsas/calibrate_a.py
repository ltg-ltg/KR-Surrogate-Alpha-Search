"""단계 A 값 산출: 리밸런싱 주기, 회전율 상한, IS 전용 임계값 조정, 마진 적정성 (설계서 8.2·8.5·11절, IS만).

    python -m krsas.calibrate_a        # krsas.stage_a 산출을 읽어 {data_root}/stage_a/calibration.json

Study 1 `src/experiment/criteria.py`·`calibrate.py`(v1.0-results)의 방법을 IS 전용 통과로 옮겼다.
    방향: IS 순 IR이 큰 쪽 (max(IR+, IR−), 동률이면 +)
    IS 전용 통과 = 기준 ① 방향 IS 순 IR ≥ h, ③ 방향 IS 평균 회전율 ≤ 상한, ④ 기존 통과와 IS 일별 순 초과수익
                   |상관| < 0.7 (IS IR 내림차순 탐욕). 기준 ②(Test)는 쓰지 않는다 (8.2절)
    회전율 상한: 그 주기에서 무작위 1,000개 수식(시그널 필터 통과 중)의 방향 IS 평균 회전율 90번째 백분위
    임계값 조정 (8.5절): 200회 시뮬레이션마다 시그널 필터 통과 수식을 무작위로 섞어 앞 W = 1,000개(warm-up)를
        빼고 다음 B = 2,000개에서 고유 통과를 센다. 평균 ≥ 22인 가장 높은 h (0.1~1.5, 0.05 간격).
        안 되면 목표 15 (폴백 ③), 그래도 안 되면 중단 (폴백 ④). 폴백 ②(Test 임계값)는 IS 전용이라 해당 없음
    주기 선택 (8.2절): 각 주기에서 조정한 임계값에서의 무작위 arm 기대 고유 통과가 가장 큰 주기, 동률이면 더 긴 주기
    마진 적정성 (11절): 마진 0.1이 무작위 1,000개 방향 IS IR 표준편차의 0.1~0.5배 안인지
난수: Study 1 `src.common.seeds.rng_for`(마스터 시드 20261101)에서 이름 "kr_stageA_*"로 파생
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import stage_a as A
from . import study1

study1.ensure()
from src.common.seeds import rng_for  # noqa: E402

W, B_BUDGET, N_SIMS = 1000, 2000, 200
TARGETS = (22, 15)
GRID = np.round(np.arange(0.10, 1.50 + 1e-9, 0.05), 4)
MAX_CORR = 0.70
TURNOVER_PCT = 90
N_RANDOM = 1000
MARGIN = 0.10
PERIOD_ORDER = {"daily": 0, "weekly": 1, "monthly": 2}      # 동률이면 긴 주기


def directed(res: pd.DataFrame, period: str) -> pd.DataFrame:
    pos, neg = res[f"{period}_ir_pos"].to_numpy(), res[f"{period}_ir_neg"].to_numpy()
    dirn = np.where(pos >= neg, 1, -1)
    pick = lambda k: np.where(dirn == 1, res[f"{period}_{k}_pos"], res[f"{period}_{k}_neg"])
    return pd.DataFrame({"formula_id": res["formula_id"].to_numpy(), "direction": dirn, "ir": np.maximum(pos, neg),
                         "turnover": pick("turnover"), "te": pick("te_ann"), "excess": pick("excess_ann")})


class ExcessStore:
    """방향을 적용한 IS 일별 순 초과수익을 표준화해 상관을 계산한다."""

    def __init__(self, arr: np.ndarray, direction: np.ndarray):
        self.arr = arr                       # (2, n, days) float32, 0 = +, 1 = −
        self.k = np.where(direction == 1, 0, 1)

    def standardized(self, idx: np.ndarray) -> np.ndarray:
        x = self.arr[self.k[idx], idx].astype(np.float64)
        x = np.nan_to_num(x)
        x = x - x.mean(axis=1, keepdims=True)
        sd = np.sqrt((x ** 2).mean(axis=1, keepdims=True))
        return np.divide(x, sd, out=np.zeros_like(x), where=sd > 0)


def greedy_unique(idx: np.ndarray, ir: np.ndarray, store: ExcessStore, max_corr: float = MAX_CORR) -> np.ndarray:
    order = np.lexsort((idx, -ir[idx]))
    adopted = np.zeros(len(idx), dtype=bool)
    if len(idx) == 0:
        return adopted
    z = store.standardized(idx[order])
    n = z.shape[1]
    kept = np.empty_like(z)
    k = 0
    for i in range(len(order)):
        if k and (np.abs(kept[:k] @ z[i]) / n).max() >= max_corr:
            continue
        kept[k] = z[i]
        k += 1
        adopted[order[i]] = True
    return adopted


def unique_by_threshold(sel, d, store, cap) -> np.ndarray:
    ir, to = d["ir"].to_numpy(), d["turnover"].to_numpy()
    cand = sel[(ir[sel] >= GRID.min()) & (to[sel] <= cap)]
    adopted = cand[greedy_unique(cand, ir, store)]      # 엄격한 임계값의 채택 집합은 접두사 (IR 내림차순)
    a = ir[adopted]
    return np.array([(a >= h).sum() for h in GRID])


def calibrate_period(res: pd.DataFrame, period: str) -> dict:
    d = directed(res, period)
    ids, arr = A.load_excess(period)
    if not np.array_equal(ids, d["formula_id"].to_numpy()):
        raise RuntimeError("결과와 초과수익 배열의 수식 순서가 다릅니다")
    store = ExcessStore(arr, d["direction"].to_numpy())
    eval_idx = np.flatnonzero(res["filter_pass"].to_numpy())
    rnd = rng_for("kr_stageA_random1000").choice(eval_idx, N_RANDOM, replace=False)
    cap = float(np.percentile(d["turnover"].to_numpy()[rnd], TURNOVER_PCT))
    ir_sd = float(d["ir"].to_numpy()[rnd].std(ddof=1))
    orders = [rng_for("kr_stageA_calibration", period, s).permutation(eval_idx)[W:W + B_BUDGET]
              for s in range(N_SIMS)]
    counts = np.stack([unique_by_threshold(o, d, store, cap) for o in orders])
    mean = counts.mean(axis=0)
    chosen = None
    for step, need in enumerate(TARGETS, start=1):
        ok = np.flatnonzero(mean >= need)
        if len(ok):
            j = int(ok.max())
            chosen = {"fallback_step": "기본" if step == 1 else "③ 목표 15", "target": need,
                      "is_ir_threshold": float(GRID[j]), "random_unique_mean": float(mean[j])}
            break
    rand = d.iloc[rnd]
    return {
        "period": period, "n_eval": int(len(eval_idx)), "turnover_cap": cap,
        "random1000": {"ir_mean": float(rand["ir"].mean()), "ir_sd": ir_sd,
                       "ir_pct": {str(q): float(np.percentile(rand["ir"], q)) for q in (5, 25, 50, 75, 95)},
                       "turnover_pct": {str(q): float(np.percentile(rand["turnover"], q)) for q in (10, 50, 90)},
                       "share_dir_pos": float((rand["direction"] == 1).mean())},
        "margin_ratio": MARGIN / ir_sd if ir_sd > 0 else None,
        "margin_ok": bool(ir_sd > 0 and 0.1 <= MARGIN / ir_sd <= 0.5),
        "random_unique_by_threshold": {f"{h:.2f}": {"mean": round(float(m), 2), "p5": float(np.percentile(c, 5)),
                                                     "min": int(c.min())}
                                       for h, m, c in zip(GRID, mean, counts.T)},
        "chosen": chosen,
        "pool_base_rate_at_chosen": (None if chosen is None else float(
            ((d["ir"].to_numpy()[eval_idx] >= chosen["is_ir_threshold"]) &
             (d["turnover"].to_numpy()[eval_idx] <= cap)).mean())),
    }


def run() -> dict:
    res = A.load_results()
    out = {"rule": {"W": W, "B": B_BUDGET, "n_sims": N_SIMS, "targets": TARGETS, "grid": [float(GRID[0]),
            float(GRID[-1]), 0.05], "max_corr": MAX_CORR, "turnover_pct": TURNOVER_PCT, "margin": MARGIN},
           "n_formulas": int(len(res)), "filter": res["filter_reason"].value_counts().to_dict(), "periods": {}}
    for per in A.PERIODS:
        out["periods"][per] = calibrate_period(res, per)
    ok = {p: v for p, v in out["periods"].items() if v["chosen"]}
    if ok:
        best = max(ok, key=lambda p: (round(ok[p]["chosen"]["random_unique_mean"], 6), PERIOD_ORDER[p]))
        out["selected_period"] = best
    else:
        out["selected_period"] = None
        out["status"] = "폴백 ④: 모든 주기에서 무작위 arm 기대 고유 통과가 15 미만 → 설계 재검토 (설계서 4절)"
    (A.out_dir() / "calibration.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def main() -> None:
    out = run()
    for p, v in out["periods"].items():
        print(p, "회전율 상한", round(v["turnover_cap"], 3), "| 무작위 IR sd", round(v["random1000"]["ir_sd"], 3),
              "마진 비율", round(v["margin_ratio"] or 0, 2), "|", v["chosen"])
    print("선택 주기:", out["selected_period"])


if __name__ == "__main__":
    main()
