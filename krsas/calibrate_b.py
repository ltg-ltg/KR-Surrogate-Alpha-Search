"""단계 B-2: 임계값 조정 (설계서 8.5절, 사전 등록 2절 "임계값 조정"). 무작위 arm만으로, IS·Test만.

    python -m krsas.calibrate_b                  # 주 결과 → {data_root}/stage_b/thresholds_main.json
    python -m krsas.calibrate_b --variant sens   # 민감도 유니버스에서 다시 정한 값 (12절)

방향 = 그 변형의 IS 순 IR이 큰 쪽. 통과 = ① 방향 IS IR ≥ h ② 같은 방향 Test IR ≥ Test 임계값
③ 방향 IS 회전율 ≤ 0.792 ④ IS 일별 순 초과수익 |상관| < 0.7 (IS IR 내림차순 탐욕, 평가 집합마다).
200회 × (평가 집합을 섞어 앞 1,000개 warm-up 제외, 다음 2,000개). 평균 고유 통과 ≥ 22인 가장 높은 h
(0.10~1.50, 0.05 간격). 폴백: ② Test 0.3 → 0.0, ③ 목표 22 → 15, ④ 중단. 상관 0.7은 완화하지 않는다.
평가 집합 = 주 유니버스 IS 시그널 필터 통과. 난수 rng_for("kr_B_calibration", 변형, s)
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from . import calibrate_a as CA
from . import stage_b as SB
from . import study1

study1.ensure()
from src.common.seeds import rng_for  # noqa: E402

TO_CAP = 0.792
TEST_THR, TEST_FALLBACK = 0.3, 0.0
TARGET, TARGET_FALLBACK = 22, 15
W, B, N_SIMS = 1000, 2000, 200
GRID = CA.GRID


def directed(res: pd.DataFrame, variant: str = "main") -> pd.DataFrame:
    g = lambda per, k, s: res[f"{variant}_{per}_{k}_{s}"].to_numpy()
    pos, neg = g("is", "ir", "pos"), g("is", "ir", "neg")
    dirn = np.where(pos >= neg, 1, -1)
    pick = lambda per, k: np.where(dirn == 1, g(per, k, "pos"), g(per, k, "neg"))
    return pd.DataFrame({"formula_id": res["formula_id"].to_numpy(), "direction": dirn, "ir": np.maximum(pos, neg),
                         "test_ir": pick("test", "ir"), "turnover": pick("is", "turnover")})


def pass_1to3(d: pd.DataFrame, idx: np.ndarray, is_thr: float, test_thr: float) -> np.ndarray:
    return ((d["ir"].to_numpy()[idx] >= is_thr) & (d["test_ir"].to_numpy()[idx] >= test_thr)
            & (d["turnover"].to_numpy()[idx] <= TO_CAP))


def unique_by_threshold(sel, d, store, test_thr) -> np.ndarray:
    cand = sel[pass_1to3(d, sel, GRID.min(), test_thr)]
    ir = d["ir"].to_numpy()
    adopted = cand[CA.greedy_unique(cand, ir, store)]
    a = ir[adopted]
    return np.array([(a >= h).sum() for h in GRID])


def count_unique(sel, d, store, is_thr, test_thr) -> tuple[int, np.ndarray]:
    cand = sel[pass_1to3(d, sel, is_thr, test_thr)]
    adopted = cand[CA.greedy_unique(cand, d["ir"].to_numpy(), store)]
    return int(len(adopted)), adopted


def calibrate(variant: str = "main") -> dict:
    res = SB.load_results()
    d = directed(res, variant)
    store = CA.ExcessStore(SB.load_is_excess(variant), d["direction"].to_numpy())
    eval_idx = np.flatnonzero(res["filter_pass"].to_numpy())
    orders = [rng_for("kr_B_calibration", variant, s).permutation(eval_idx)[W:W + B] for s in range(N_SIMS)]
    out = {"variant": variant, "n_eval": int(len(eval_idx)), "steps": []}
    chosen = None
    plan = [("기본", TEST_THR, TARGET), ("② Test 0.0", TEST_FALLBACK, TARGET),
            ("③ 목표 15", TEST_FALLBACK, TARGET_FALLBACK)]
    cache = {}
    for name, test_thr, need in plan:
        if test_thr not in cache:
            cache[test_thr] = np.stack([unique_by_threshold(o, d, store, test_thr) for o in orders])
        counts = cache[test_thr]
        mean = counts.mean(axis=0)
        out["steps"].append({"step": name, "test_threshold": test_thr, "target": need,
                             "random_unique": {f"{h:.2f}": {"mean": round(float(m), 2),
                                                            "p5": float(np.percentile(c, 5))}
                                               for h, m, c in zip(GRID, mean, counts.T)}})
        ok = np.flatnonzero(mean >= need)
        if len(ok):
            j = int(ok.max())
            chosen = {"step": name, "is_ir": float(GRID[j]), "test_ir": test_thr, "target": need,
                      "random_unique_mean": float(mean[j])}
            break
    if chosen is None:
        out["status"] = "폴백 ④: 중단·보고 (설계서 8.5절)"
    else:
        out["status"] = "ok"
        out["thresholds"] = {**chosen, "max_turnover": TO_CAP, "max_corr": CA.MAX_CORR}
        n_pool, _ = count_unique(eval_idx, d, store, chosen["is_ir"], chosen["test_ir"])
        out["pool_unique_passes"] = n_pool
        out["pool_pass_1to3_rate"] = float(pass_1to3(d, eval_idx, chosen["is_ir"], chosen["test_ir"]).mean())
        out["random_baseline_unique_at_B"] = chosen["random_unique_mean"]
    (SB.out_dir() / f"thresholds_{variant}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                                             encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="main")
    out = calibrate(parser.parse_args().variant)
    print(out["status"], out.get("thresholds"), "| 풀 고유 통과", out.get("pool_unique_passes"))


if __name__ == "__main__":
    main()
