"""단계 A 검정력과 시드 수 (설계서 11절, IS만).

    python -m krsas.power_a --workers 5        # 시드 10개, {data_root}/stage_a/power/
    python -m krsas.power_a --summary          # 결과로 시드 수 계산 → power.json

설계서 11절 "검정력과 시드 수":
  개수(R1, R2): 시드 10개로 주 칸과 비교 상대를 IS 전용 통과(기준 ①③④, 월간, IS IR ≥ 0.50, 회전율 ≤ 0.792)로
    돌려 시드 대응 차이의 표준편차를 추정한다. Study 1(미국)에서 관찰한 차이의 절반을 해당 Bonferroni 수준
    (R1 m = 1 → 양측 0.05, R2 m = 2 → 양측 0.025)에서 80% 검정력으로 잡는 시드 수 n = ((z_a + z_0.8) sd / δ)².
    δ: R1 = (24.1 − 14.9) / 2 = 4.6, R2 ① = (25.8 − 14.9) / 2 = 5.45, R2 ② = (35.6 − 21.8) / 2 = 6.9
  품질(R3, S-b): IS 가짜 분할. 앞(2011~2014) IS로 타깃·통과 판정, 뒤(2015~2017)를 가짜 OOS로 쓴다.
    R3: 시드별 채택 시그널 가짜 OOS IR 중앙값의 차이(주 칸 − 무작위) → 95% 반폭(n) = t × sd / √n
    S-b: 보정 회귀(가짜 OOS IR ~ 앞 IS IR + 잔차 더미)의 더미 계수, 고유 시그널 단위, 시드 부트스트랩
         97.5% 구간 반폭을 시드 10개에서 재고 √(10 / n)로 늘린다
  시드 수 = max(개수, 품질 R3), 하한 30. S-b 반폭이 상한 시드 수에서도 0.07을 넘으면 S-b를 보조로 내린다

칸 (설계서 3.1절 주 칸): random, lgbm_base_topb_{raw,resid}, lgbm_ext_mmr_{raw,resid} (LightGBM, Optuna 50 trials,
스켈레톤 GroupKFold 5, Study 1 `tune_and_fit`, MMR λ = 0.5는 Study 1 `mmr_select`, 유사도는 기본 스키마)
잔차 타깃 (9.1절): warm-up 중 기준 ①③ 통과를 IS IR 내림차순 탐욕(상관 < 0.7)으로 K = 5개, 모자라면 기준 ①을
빼고 채운다. 타깃 = 대표 K개의 일별 초과수익에 OLS한 잔차(절편 포함)의 IR
가짜 분할의 회전율 기준은 IS 전체 회전율로 대신한다 (구간별 회전율을 저장하지 않았다)
난수: Study 1 rng_for, 이름 "kr_stageA_power_*" (사전 등록 뒤 본 실험의 시드 이름과 겹치지 않는다)
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import calibrate_a as C
from . import stage_a as A
from . import study1

study1.ensure()
from src.common.seeds import derive_seed, rng_for  # noqa: E402

PERIOD = "monthly"
IS_THR, TO_CAP = 0.50, 0.792          # docs/stageA_values.md
FRONT_END = "2014-12-31"
N_SEEDS, W, B, K_REF, LAM = 10, 1000, 2000, 5, 0.5
TRIALS, FOLDS = 50, 5
ARMS = ("random", "lgbm_base_topb_raw", "lgbm_base_topb_resid", "lgbm_ext_mmr_raw", "lgbm_ext_mmr_resid")
DELTA = {"R1": 4.6, "R2_base_topb": 5.45, "R2_ext_mmr": 6.9}
_W: dict = {}


def pdir() -> Path:
    return A.out_dir() / "power"


# ---------------------------------------------------------------- 준비 (한 번)
def prepare() -> None:
    """피처, 방향 적용 지표, 초과수익(mmap용 .npy)을 만든다."""
    from src.features.ast_features import build_features, freeze_schema
    from src.generator.grammar import load_grammar

    from .pool import load_pool
    d = pdir()
    d.mkdir(parents=True, exist_ok=True)
    pool = load_pool()
    g = load_grammar()
    for ext, name in ((False, "base"), (True, "ext")):
        f = d / f"features_{name}.npy"
        if not f.exists():
            df = build_features(pool["formula"].tolist(), g, ext)
            keep, _ = freeze_schema(df)
            np.save(f, df[keep].to_numpy(np.float64))
            (d / f"schema_{name}.json").write_text(json.dumps(keep), encoding="utf-8")
    ex = d / "excess_monthly.npy"
    if not ex.exists():
        ids, arr = A.load_excess(PERIOD)
        if not np.array_equal(ids, pool["formula_id"].to_numpy()):
            raise RuntimeError("초과수익 배열 순서가 풀과 다릅니다")
        np.save(ex, arr)                                     # (2, 40000, days) float32
    res = A.load_results()
    meta = json.loads((A.out_dir() / "meta.json").read_text(encoding="utf-8"))
    dates = pd.DatetimeIndex(meta["is_dates"])
    front = np.asarray(dates <= pd.Timestamp(FRONT_END))
    arr = np.load(ex, mmap_mode="r")
    ir = lambda x: np.nanmean(x, axis=1) / np.nanstd(x, axis=1, ddof=1) * np.sqrt(252)
    rows = {}
    for k, name in ((0, "pos"), (1, "neg")):
        rows[f"front_{name}"] = np.concatenate([ir(arr[k, i:i + 2000][:, front]) for i in range(0, arr.shape[1], 2000)])
        rows[f"back_{name}"] = np.concatenate([ir(arr[k, i:i + 2000][:, ~front]) for i in range(0, arr.shape[1], 2000)])
    full = C.directed(res, PERIOD)
    sub = pd.DataFrame({"formula_id": res["formula_id"], "filter_pass": res["filter_pass"],
                        "skeleton": pool["skeleton"], "ir": full["ir"], "direction": full["direction"],
                        "turnover": full["turnover"], **rows})
    fdir = np.where(sub["front_pos"] >= sub["front_neg"], 1, -1)
    sub["front_direction"] = fdir
    sub["front_ir"] = np.where(fdir == 1, sub["front_pos"], sub["front_neg"])
    sub["back_ir"] = np.where(fdir == 1, sub["back_pos"], sub["back_neg"])
    sub.to_parquet(d / "directed.parquet", index=False)
    np.save(d / "front_mask.npy", front)


# ---------------------------------------------------------------- 시드 하나
def _init() -> None:
    d = pdir()
    _W["d"] = pd.read_parquet(d / "directed.parquet")
    _W["arr"] = np.load(d / "excess_monthly.npy", mmap_mode="r")
    _W["front"] = np.load(d / "front_mask.npy")
    _W["X"] = {n: np.load(d / f"features_{n}.npy") for n in ("base", "ext")}
    from src.experiment.mmr import unit_feature_vectors
    _W["U"] = unit_feature_vectors(_W["X"]["base"])


class _Store(C.ExcessStore):
    """구간(cols)과 방향을 고른 초과수익."""

    def __init__(self, arr, direction, cols):
        super().__init__(arr, direction)
        self.cols = cols

    def rows(self, idx):
        x = np.stack([self.arr[k, i] for k, i in zip(self.k[idx], idx)]).astype(np.float64)
        return np.nan_to_num(x[:, self.cols])

    def standardized(self, idx):
        x = self.rows(idx)
        x = x - x.mean(axis=1, keepdims=True)
        sd = np.sqrt((x ** 2).mean(axis=1, keepdims=True))
        return np.divide(x, sd, out=np.zeros_like(x), where=sd > 0)


def residual_targets(warm, ir, turnover, store) -> tuple[np.ndarray, int]:
    """9.1절 대표 K와 잔차 IR."""
    def pick(ok_mask, chosen):
        cand = warm[ok_mask & ~np.isin(warm, chosen)]
        cand = cand[np.lexsort((cand, -ir[cand]))]
        for c in cand:
            if len(chosen) == K_REF:
                break
            if chosen:
                zc = store.standardized(np.array(chosen))
                if (np.abs(zc @ store.standardized(np.array([c]))[0]) / zc.shape[1]).max() >= C.MAX_CORR:
                    continue
            chosen.append(int(c))
        return chosen
    refs = pick((ir[warm] >= IS_THR) & (turnover[warm] <= TO_CAP), [])
    if len(refs) < K_REF:
        refs = pick(turnover[warm] <= TO_CAP, refs)
    net = store.rows(warm)
    if refs:
        F = store.rows(np.array(refs)).T
        X = np.column_stack([np.ones(len(F)), F])
        coef, *_ = np.linalg.lstsq(X, net.T, rcond=None)
        resid = net - (F @ coef[1:]).T
    else:
        resid = net
    sd = resid.std(axis=1, ddof=1)
    y = np.divide(resid.mean(axis=1), sd, out=np.zeros(len(sd)), where=sd > 1e-12) * np.sqrt(252)
    return y, len(refs)


def _seed(args) -> dict:
    from src.experiment.mmr import mmr_select, percentile_rank
    from src.surrogate.tune import tune_and_fit
    seed, mode = args
    d, arr, front = _W["d"], _W["arr"], _W["front"]
    eval_idx = np.flatnonzero(d["filter_pass"].to_numpy())
    warm = np.sort(rng_for("kr_stageA_power_warmup", seed).choice(eval_idx, W, replace=False))
    cand = np.setdiff1d(eval_idx, warm)
    if mode == "full":
        ir, dirn, cols = d["ir"].to_numpy(), d["direction"].to_numpy(), np.ones(arr.shape[2], bool)
    else:
        ir, dirn, cols = d["front_ir"].to_numpy(), d["front_direction"].to_numpy(), front
    turnover = d["turnover"].to_numpy()
    store = _Store(arr, dirn, cols)
    groups = d["skeleton"].to_numpy()
    y_raw = ir[warm]
    y_res, k_used = residual_targets(warm, ir, turnover, store)
    out = {"seed": seed, "mode": mode, "k_refs": k_used, "arms": {}}
    for arm in ARMS:
        t0 = time.perf_counter()
        if arm == "random":
            picks = rng_for("kr_stageA_power_random", seed).permutation(cand)[:B]
            info = {}
        else:
            _, schema, sel, target = arm.split("_")
            X = _W["X"][schema]
            y = y_raw if target == "raw" else y_res
            model, params, cv = tune_and_fit("lightgbm", X[warm], y, groups[warm], TRIALS, FOLDS,
                                             derive_seed("kr_stageA_power_model", seed, mode, arm))
            pred = np.asarray(model.predict(X[cand]), dtype=np.float64)
            if sel == "topb":
                picks = cand[np.lexsort((cand, -pred))][:B]
            else:
                picks = mmr_select(cand, percentile_rank(pred), _W["U"], LAM, B)
            info = {"cv_rmse": cv}
        ok = picks[(ir[picks] >= IS_THR) & (turnover[picks] <= TO_CAP)]
        adopted = ok[C.greedy_unique(ok, ir, store)]
        out["arms"][arm] = {"unique": int(len(adopted)), "adopted": adopted.tolist(),
                            "sec": round(time.perf_counter() - t0, 1), **info}
    return out


def run(workers: int) -> None:
    prepare()
    d = pdir() / "seeds"
    d.mkdir(parents=True, exist_ok=True)
    todo = [(s, m) for m in ("full", "split") for s in range(N_SEEDS)
            if not (d / f"{m}_seed{s:02d}.json").exists()]
    print(f"남은 시드×모드 {len(todo)}개, 워커 {workers}", flush=True)
    with ProcessPoolExecutor(workers, initializer=_init) as ex:
        for res in ex.map(_seed, todo):
            (d / f"{res['mode']}_seed{res['seed']:02d}.json").write_text(json.dumps(res), encoding="utf-8")
            print(f"  {res['mode']} 시드 {res['seed']}: " +
                  ", ".join(f"{a} {v['unique']}" for a, v in res["arms"].items()), flush=True)


# ---------------------------------------------------------------- 요약
def _n_seeds(sd: float, delta: float, alpha: float) -> float:
    from scipy.stats import norm
    return float(((norm.ppf(1 - alpha / 2) + norm.ppf(0.8)) * sd / delta) ** 2)


def summarize() -> dict:
    from scipy.stats import t as tdist
    d = pd.read_parquet(pdir() / "directed.parquet")
    load = lambda m: [json.loads(f.read_text(encoding="utf-8")) for f in sorted((pdir() / "seeds").glob(f"{m}_*.json"))]
    full, split = load("full"), load("split")
    cnt = pd.DataFrame([{a: r["arms"][a]["unique"] for a in ARMS} for r in full])
    diffs = {"R1": cnt["lgbm_base_topb_raw"] - cnt["random"],
             "R2_base_topb": cnt["lgbm_base_topb_resid"] - cnt["lgbm_base_topb_raw"],
             "R2_ext_mmr": cnt["lgbm_ext_mmr_resid"] - cnt["lgbm_ext_mmr_raw"]}
    alpha = {"R1": 0.05, "R2_base_topb": 0.025, "R2_ext_mmr": 0.025}
    count = {k: {"mean_diff": float(v.mean()), "sd": float(v.std(ddof=1)), "delta": DELTA[k], "alpha": alpha[k],
                 "n_needed": _n_seeds(float(v.std(ddof=1)), DELTA[k], alpha[k])} for k, v in diffs.items()}
    back, front_ir = d["back_ir"].to_numpy(), d["front_ir"].to_numpy()
    med = pd.DataFrame([{a: float(np.median(back[r["arms"][a]["adopted"]])) if r["arms"][a]["adopted"] else np.nan
                         for a in ARMS} for r in split])
    r3 = (med["lgbm_base_topb_raw"] - med["random"]).dropna()
    sd3 = float(r3.std(ddof=1))
    hw = lambda n, conf: float(tdist.ppf(1 - (1 - conf) / 2, n - 1) * sd3 / np.sqrt(n))
    # S-b: 고유 시그널 단위 보정 회귀, 시드 부트스트랩
    rng = np.random.default_rng(0)

    def coef(seeds, raw_arm, res_arm):
        rows = []
        for arm, dummy in ((raw_arm, 0.0), (res_arm, 1.0)):
            ids = sorted({i for s in seeds for i in split[s]["arms"][arm]["adopted"]})
            rows += [(front_ir[i], dummy, back[i]) for i in ids]
        a = np.array(rows)
        X = np.column_stack([np.ones(len(a)), a[:, 0], a[:, 1]])
        return float(np.linalg.lstsq(X, a[:, 2], rcond=None)[0][2])
    sb = {}
    for cell, (raw_arm, res_arm) in {"base_topb": ("lgbm_base_topb_raw", "lgbm_base_topb_resid"),
                                     "ext_mmr": ("lgbm_ext_mmr_raw", "lgbm_ext_mmr_resid")}.items():
        boots = [coef(rng.integers(0, len(split), len(split)), raw_arm, res_arm) for _ in range(2000)]
        lo, hi = np.percentile(boots, [1.25, 98.75])
        hw10 = float((hi - lo) / 2)
        sb[cell] = {"coef": coef(range(len(split)), raw_arm, res_arm), "halfwidth_975_at_10": hw10,
                    "halfwidth_975_at_30": hw10 * np.sqrt(len(split) / 30),
                    "seeds_for_0.07": float(len(split) * (hw10 / 0.07) ** 2)}
    n_count = max(v["n_needed"] for v in count.values())
    out = {"count": count, "count_means": cnt.mean().round(2).to_dict(),
           "quality_R3": {"mean_diff_median_back_ir": float(r3.mean()), "sd": sd3,
                          "halfwidth_95": {str(n): hw(n, 0.95) for n in (10, 30, 50, 100)}},
           "S_b": sb, "k_refs_full": [r["k_refs"] for r in full],
           "seed_count": {"count_rule": n_count, "floor": 30, "chosen": int(max(30, np.ceil(n_count)))},
           "note": "품질 R3은 반폭만 보고한다 (마진 0.1 대비 판정 가능성). 시드 수 계산식에는 개수 규칙과 하한만 쓴다"}
    (A.out_dir() / "power.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    if args.summary:
        print(json.dumps(summarize(), ensure_ascii=False, indent=1))
    else:
        run(args.workers)


if __name__ == "__main__":
    main()
