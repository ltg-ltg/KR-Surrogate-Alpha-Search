"""단계 B-3: 153시드 리플레이 (설계서 9절, 사전 등록 1·2·3절). IS·Test만 쓴다.

    python -m krsas.replay_b --workers 6            # 남은 시드만
    python -m krsas.replay_b --seeds 0 1 --workers 2

시드 s마다 (같은 warm-up을 모든 arm이 공유)
    warm-up: rng_for("kr_B_warmup", s)로 평가 집합에서 1,000개. 후보 = 평가 집합 − warm-up
    모델 칸: {lightgbm, xgboost} × {base, ext} × {raw, resid} 학습 8번 (Study 1 tune_and_fit, Optuna 50 trials,
             스켈레톤 GroupKFold 5, 시드 derive_seed("kr_B_model", s, 칸)). 선택: 상위 B(topb), MMR λ 0.5(mmr05).
             원시 타깃은 D1 사다리 λ 0.25(mmr025)·1.0(mmr10)을 더한다. 유사도는 기본 스키마 (사전 등록 3.1절)
    잔차 타깃: 사전 등록 3.2절 (대표 K = 5, IS 임계값 = B-2 조정값, 회전율 ≤ 0.792)
    기준선: random (rng_for("kr_B_random", s) 순서 앞 2,000), struct_random (무작위 관련도 + MMR λ 0.5, Study 1과 같음),
            prefilter (무작위 순서로 평가하되 이미 통과한 시그널과 같은 스켈레톤은 건너뜀, 건너뛴 것은 예산 밖),
            heuristic (Study 1 ComponentHeuristic m = 5, 원시 타깃, 상위 B)
            같은 총 예산 무작위 3,000은 시드마다 rng_for("kr_B_random3000", s)로 평가 집합 전체에서 뽑는다
    평가: 칸마다 2,000개 선택 안에서 기준 ①~④ 고유 통과 (B-2 임계값). warm-up은 세지 않는다
산출: {data_root}/stage_b/replay/seed_XXX.json (칸별 고유 통과·채택 목록·학습 정보), picks_XXX.npz (선택 순서)
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import calibrate_a as CA
from . import calibrate_b as CB
from . import stage_b as SB
from . import study1

study1.ensure()
from src.common.seeds import derive_seed, rng_for  # noqa: E402

N_SEEDS, W, B, K_REF = 153, 1000, 2000, 5
TRIALS, FOLDS = 50, 5
MODELS, SCHEMAS, TARGETS = ("lightgbm", "xgboost"), ("base", "ext"), ("raw", "resid")
LAMS = {"topb": None, "mmr025": 0.25, "mmr05": 0.5, "mmr10": 1.0}
_W: dict = {}


def rdir() -> Path:
    return SB.out_dir() / "replay"


def prepare() -> dict:
    """한 번: 방향 적용 지표, mmap 초과수익, 피처, 임계값."""
    from src.features.ast_features import build_features, freeze_schema
    from src.generator.grammar import load_grammar

    from .pool import load_pool
    d = rdir()
    d.mkdir(parents=True, exist_ok=True)
    th = json.loads((SB.out_dir() / "thresholds_main.json").read_text(encoding="utf-8"))
    if th.get("status") != "ok":
        raise RuntimeError(f"B-2 임계값이 없습니다: {th.get('status')}")
    res = SB.load_results()
    pool = load_pool()
    if not np.array_equal(res["formula_id"].to_numpy(), pool["formula_id"].to_numpy()):
        raise RuntimeError("B-1 결과와 풀의 순서가 다릅니다")
    dd = CB.directed(res, "main")
    dd["filter_pass"] = res["filter_pass"].to_numpy()
    dd["skeleton"] = pool["skeleton"].to_numpy()
    dd.to_parquet(d / "directed.parquet", index=False)
    if not (d / "is_excess_main.npy").exists():
        np.save(d / "is_excess_main.npy", SB.load_is_excess("main"))
    g = load_grammar()
    for ext, name in ((False, "base"), (True, "ext")):
        if not (d / f"features_{name}.npy").exists():
            df = build_features(pool["formula"].tolist(), g, ext)
            keep, _ = freeze_schema(df)
            np.save(d / f"features_{name}.npy", df[keep].to_numpy(np.float64))
    return th["thresholds"]


def _init() -> None:
    from src.experiment.mmr import unit_feature_vectors
    from src.generator.grammar import load_grammar
    from src.surrogate.heuristic import components

    from .pool import load_pool
    d = rdir()
    _W["d"] = pd.read_parquet(d / "directed.parquet")
    _W["arr"] = np.load(d / "is_excess_main.npy", mmap_mode="r")
    _W["X"] = {n: np.load(d / f"features_{n}.npy") for n in SCHEMAS}
    _W["U"] = unit_feature_vectors(_W["X"]["base"])
    th = json.loads((SB.out_dir() / "thresholds_main.json").read_text(encoding="utf-8"))["thresholds"]
    _W["th"] = th
    g = load_grammar()
    _W["comps"] = [components(t, g) for t in load_pool()["formula"]]


class _Store(CA.ExcessStore):
    def rows(self, idx):
        return np.nan_to_num(np.stack([self.arr[k, i] for k, i in zip(self.k[idx], idx)]).astype(np.float64))

    def standardized(self, idx):
        x = self.rows(idx)
        x = x - x.mean(axis=1, keepdims=True)
        sd = np.sqrt((x ** 2).mean(axis=1, keepdims=True))
        return np.divide(x, sd, out=np.zeros_like(x), where=sd > 0)


def residual_targets(warm, ir, turnover, store, is_thr, cap) -> tuple[np.ndarray, list[int]]:
    def pick(mask, chosen):
        cand = warm[mask & ~np.isin(warm, chosen)]
        for c in cand[np.lexsort((cand, -ir[cand]))]:
            if len(chosen) == K_REF:
                break
            if chosen:
                zc = store.standardized(np.array(chosen))
                if (np.abs(zc @ store.standardized(np.array([c]))[0]) / zc.shape[1]).max() >= CA.MAX_CORR:
                    continue
            chosen.append(int(c))
        return chosen
    refs = pick((ir[warm] >= is_thr) & (turnover[warm] <= cap), [])
    if len(refs) < K_REF:
        refs = pick(turnover[warm] <= cap, refs)          # 기준 ①만 뺀다 (③은 유지)
    net = store.rows(warm)
    if refs:
        F = store.rows(np.array(refs)).T
        X = np.column_stack([np.ones(len(F)), F])
        coef, *_ = np.linalg.lstsq(X, net.T, rcond=None)
        resid = net - (F @ coef[1:]).T
    else:
        resid = net
    sd = resid.std(axis=1, ddof=1)
    return np.divide(resid.mean(axis=1), sd, out=np.zeros(len(sd)), where=sd > 1e-12) * np.sqrt(252), refs


def evaluate(picks, d, store, th) -> dict:
    n, adopted = CB.count_unique(np.asarray(picks), d, store, th["is_ir"], th["test_ir"])
    return {"unique": n, "adopted": adopted.tolist(),
            "n_pass_1to3": int(CB.pass_1to3(d, np.asarray(picks), th["is_ir"], th["test_ir"]).sum())}


def prefilter_picks(order, d, store, th) -> list[int]:
    """무작위 순서로 평가하되, 이미 통과(①~④, 평가 순서대로)한 시그널과 같은 스켈레톤은 건너뛴다."""
    sk = d["skeleton"].to_numpy()
    passed_sk, picks, adopted = set(), [], []
    ok13 = CB.pass_1to3(d, order, th["is_ir"], th["test_ir"])
    for i, ok in zip(order, ok13):
        if len(picks) == B:
            break
        if sk[i] in passed_sk:
            continue
        picks.append(int(i))
        if ok:
            if adopted:
                zc = store.standardized(np.array(adopted))
                if (np.abs(zc @ store.standardized(np.array([i]))[0]) / zc.shape[1]).max() >= CA.MAX_CORR:
                    continue
            adopted.append(int(i))
            passed_sk.add(sk[i])
    return picks


def _seed(seed: int) -> dict:
    from src.experiment.mmr import mmr_select, percentile_rank
    from src.surrogate.heuristic import ComponentHeuristic
    from src.surrogate.tune import tune_and_fit
    d, th = _W["d"], _W["th"]
    ir, turnover = d["ir"].to_numpy(), d["turnover"].to_numpy()
    store = _Store(_W["arr"], d["direction"].to_numpy())
    groups = d["skeleton"].to_numpy()
    eval_idx = np.flatnonzero(d["filter_pass"].to_numpy())
    warm = np.sort(rng_for("kr_B_warmup", seed).choice(eval_idx, W, replace=False))
    cand = np.setdiff1d(eval_idx, warm)
    y_res, refs = residual_targets(warm, ir, turnover, store, th["is_ir"], th["max_turnover"])
    out = {"seed": seed, "refs": refs, "cells": {}, "fits": {}}
    picks_all = {}

    def add(cell, picks, info=None):
        picks_all[cell] = np.asarray(picks, dtype=np.int32)
        out["cells"][cell] = {**evaluate(picks, d, store, th), **(info or {})}

    add("random", rng_for("kr_B_random", seed).permutation(cand)[:B])
    add("random3000", rng_for("kr_B_random3000", seed).permutation(eval_idx)[:W + B])
    rel = rng_for("kr_B_struct_random", seed).uniform(size=len(cand))
    add("struct_random", mmr_select(cand, rel, _W["U"], 0.5, B))
    add("prefilter", prefilter_picks(rng_for("kr_B_prefilter", seed).permutation(cand), d, store, th))
    t0 = time.perf_counter()
    h = ComponentHeuristic(m=5).fit([_W["comps"][i] for i in warm], ir[warm])
    hp = h.predict([_W["comps"][i] for i in cand])
    add("heuristic_topb", cand[np.lexsort((cand, -hp))][:B], {"sec": round(time.perf_counter() - t0, 1)})
    for model in MODELS:
        for schema in SCHEMAS:
            for target in TARGETS:
                fit = f"{model}_{schema}_{target}"
                t0 = time.perf_counter()
                X = _W["X"][schema]
                y = ir[warm] if target == "raw" else y_res
                m, params, cv = tune_and_fit(model, X[warm], y, groups[warm], TRIALS, FOLDS,
                                             derive_seed("kr_B_model", seed, fit))
                pred = np.asarray(m.predict(X[cand]), dtype=np.float64)
                out["fits"][fit] = {"cv_rmse": cv, "params": params, "sec": round(time.perf_counter() - t0, 1)}
                rel = percentile_rank(pred)
                for sel, lam in LAMS.items():
                    if target == "resid" and sel in ("mmr025", "mmr10"):
                        continue
                    picks = cand[np.lexsort((cand, -pred))][:B] if lam is None else mmr_select(cand, rel, _W["U"],
                                                                                              lam, B)
                    add(f"{model}_{schema}_{sel}_{target}", picks)
    np.savez(rdir() / f"picks_{seed:03d}.npz", **picks_all)
    return out


def run(workers: int, seeds: list[int] | None) -> None:
    prepare()
    seeds = list(range(N_SEEDS)) if seeds is None else seeds
    todo = [s for s in seeds if not (rdir() / f"seed_{s:03d}.json").exists()]
    print(f"남은 시드 {len(todo)}개, 작업자 {workers}", flush=True)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(workers, initializer=_init) as ex:
        for n, res in enumerate(ex.map(_seed, todo), start=1):
            (rdir() / f"seed_{res['seed']:03d}.json").write_text(json.dumps(res), encoding="utf-8")
            c = res["cells"]
            el = time.perf_counter() - t0
            print(f"  시드 {res['seed']}: random {c['random']['unique']}, lgbm 기본 상위B 원시 "
                  f"{c['lightgbm_base_topb_raw']['unique']} ({n}/{len(todo)}, {el / 60:.0f}분, "
                  f"남은 예상 {el / n * (len(todo) - n) / 60:.0f}분)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    args = parser.parse_args()
    run(args.workers, args.seeds)


if __name__ == "__main__":
    main()
