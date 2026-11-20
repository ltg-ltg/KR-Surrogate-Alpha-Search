"""단계 B IS·Test 결과 정리 (사전 등록 4절 B-3 뒤, B-T). OOS는 읽지 않는다.

    python -m krsas.analysis_b          # {data_root}/stage_b/analysis_is_test.json

설계서 11절 판정 규칙
    개수 (R1, R2, R4): 시드 대응 차이의 부트스트랩(시드 재표집 10,000회, 백분위) 구간. R1 95%, R2 칸마다 97.5%
        (m = 2), R4 보조 95%. 판정: 예측 방향에서 구간이 0을 넘으면 성립, 반대 방향에서 넘으면 실패, 그 외 판정 불가.
        시드 대응 Wilcoxon p값을 Holm 보정해 함께 보고한다 (판정에는 쓰지 않는다)
    R2: 사전 등록 1.1절 두 칸 묶음 규칙
    S-a: 잔차 − 원시 채택 시그널의 IS IR 중앙값 차이. 고유 시그널 단위(같은 arm에서 반복 채택은 한 번), 시드 부트스트랩
        97.5%. 구간이 0 아래면 성립
    군집 진단·D1(B 단계분): 칸 × 시드의 2,000개 선택 안 IS 일별 순 초과수익 |상관| ≥ 0.7 쌍 비율(x)과 고유 통과(y).
        원시 칸 16개(모델 2 × 스키마 2 × λ 4)로 Spearman과 x 구간(10%p)별 평균 y, 시드 부트스트랩 구간
    B-T 구간 폭 리허설: Test를 가짜 OOS로 R3·R5·S-b 반폭. 채택에 기준 ②(Test ≥ 0.3)가 들어가 편향이 있다
난수: rng_for("kr_B_analysis", 이름)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from . import replay_b as R
from . import stage_b as SB
from . import study1

study1.ensure()
from src.common.seeds import rng_for  # noqa: E402

N_BOOT = 10_000
MAIN_R1 = ("lightgbm_base_topb_raw", "random")
R2_CELLS = {"base_topb": ("lightgbm_base_topb_resid", "lightgbm_base_topb_raw"),
            "ext_mmr": ("lightgbm_ext_mmr05_resid", "lightgbm_ext_mmr05_raw")}
R4 = ("lightgbm_ext_mmr05_resid", "lightgbm_base_topb_resid")
D1_CELLS = [f"{m}_{s}_{sel}_raw" for m in R.MODELS for s in R.SCHEMAS for sel in R.LAMS]
MARGIN = 0.1


def load_seeds() -> list[dict]:
    files = sorted(R.rdir().glob("seed_*.json"))
    return [json.loads(f.read_text(encoding="utf-8")) for f in files]


def boot_ci(x: np.ndarray, conf: float, name: str) -> tuple[float, float]:
    rng = rng_for("kr_B_analysis", name)
    idx = rng.integers(0, len(x), (N_BOOT, len(x)))
    m = x[idx].mean(axis=1)
    a = (1 - conf) / 2
    return float(np.quantile(m, a)), float(np.quantile(m, 1 - a))


def verdict_direction(lo, hi, expect: str) -> str:
    if expect == "neg":
        return "성립" if hi < 0 else ("실패" if lo > 0 else "판정 불가")
    return "성립" if lo > 0 else ("실패" if hi < 0 else "판정 불가")


def holm(ps: dict) -> dict:
    keys = sorted(ps, key=lambda k: ps[k])
    out, prev = {}, 0.0
    for i, k in enumerate(keys):
        v = min(1.0, max(prev, (len(keys) - i) * ps[k]))
        out[k], prev = v, v
    return out


def count_table(seeds) -> pd.DataFrame:
    return pd.DataFrame([{c: v["unique"] for c, v in s["cells"].items()} | {"seed": s["seed"]} for s in seeds]) \
        .set_index("seed").sort_index()


def count_tests(cnt: pd.DataFrame) -> dict:
    out = {}
    comps = {"R1": (*MAIN_R1, 0.95, "neg")}
    comps |= {f"R2_{k}": (*v, 0.975, "pos") for k, v in R2_CELLS.items()}
    comps["R4"] = (*R4, 0.95, "pos")
    ps = {}
    for name, (a, b, conf, expect) in comps.items():
        dlt = (cnt[a] - cnt[b]).to_numpy(float)
        lo, hi = boot_ci(dlt, conf, name)
        p = float(wilcoxon(dlt).pvalue) if np.any(dlt != 0) else 1.0
        ps[name] = p
        out[name] = {"cell": a, "vs": b, "mean_a": float(cnt[a].mean()), "mean_b": float(cnt[b].mean()),
                     "mean_diff": float(dlt.mean()), "ci": [lo, hi], "conf": conf, "expect": expect,
                     "verdict": verdict_direction(lo, hi, expect), "wilcoxon_p": p}
    for k, v in holm(ps).items():
        out[k]["wilcoxon_p_holm"] = v
    v = [out[f"R2_{k}"]["verdict"] for k in R2_CELLS]
    out["R2"] = {"verdict": "실패" if "실패" in v else ("성립" if "성립" in v else "판정 불가"),
                 "cells": dict(zip(R2_CELLS, v)), "rule": "사전 등록 1.1절"}
    return out


def _pooled(seeds, idx, arm, value):
    ids = sorted({i for k in idx for i in seeds[k]["cells"][arm]["adopted"]})
    return value[ids] if ids else np.array([])


def s_a(seeds, ir) -> dict:
    out = {}
    for cell, (res, raw) in R2_CELLS.items():
        full = range(len(seeds))
        est = float(np.median(_pooled(seeds, full, res, ir)) - np.median(_pooled(seeds, full, raw, ir)))
        rng = rng_for("kr_B_analysis", f"S-a_{cell}")
        boots = []
        for _ in range(2000):
            idx = rng.integers(0, len(seeds), len(seeds))
            a, b = _pooled(seeds, idx, res, ir), _pooled(seeds, idx, raw, ir)
            if len(a) and len(b):
                boots.append(np.median(a) - np.median(b))
        lo, hi = np.quantile(boots, [0.0125, 0.9875])
        overlap = len(set(i for s in seeds for i in s["cells"][res]["adopted"]) &
                      set(i for s in seeds for i in s["cells"][raw]["adopted"]))
        out[cell] = {"median_diff": est, "ci_975": [float(lo), float(hi)],
                     "verdict": "성립" if hi < 0 else ("깨짐" if lo > 0 else "판정 불가"),
                     "unique_signals": {res: int(len(_pooled(seeds, full, res, ir))),
                                        raw: int(len(_pooled(seeds, full, raw, ir)))}, "overlap": overlap}
    return out


def high_corr_fraction(picks: np.ndarray, store) -> float:
    z = store.standardized(np.sort(picks))
    c = np.abs(z @ z.T) / z.shape[1]
    iu = np.triu_indices(len(z), 1)
    return float((c[iu] >= 0.7).mean())


def clusters(seeds, store) -> pd.DataFrame:
    rows = []
    for s in seeds:
        picks = np.load(R.rdir() / f"picks_{s['seed']:03d}.npz")
        for cell in [*D1_CELLS, "random", "lightgbm_base_topb_resid", "lightgbm_ext_mmr05_resid"]:
            rows.append({"seed": s["seed"], "cell": cell, "x": high_corr_fraction(picks[cell], store),
                         "y": s["cells"][cell]["unique"]})
    return pd.DataFrame(rows)


def d1(cl: pd.DataFrame) -> dict:
    raw = cl[cl["cell"].isin(D1_CELLS)]
    rho = float(spearmanr(raw["x"], raw["y"]).statistic)
    rng = rng_for("kr_B_analysis", "D1")
    seeds = raw["seed"].unique()
    boots = []
    for _ in range(2000):
        pick = rng.choice(seeds, len(seeds))
        b = pd.concat([raw[raw["seed"] == s] for s in pick])
        boots.append(spearmanr(b["x"], b["y"]).statistic)
    bins = (raw["x"] // 0.1 * 0.1).round(1)
    return {"spearman": rho, "ci_95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
            "mean_y_by_x_bin": raw.groupby(bins)["y"].mean().round(2).to_dict(),
            "x_range": [float(raw["x"].min()), float(raw["x"].max())],
            "random_baseline_y": float(cl.loc[cl["cell"] == "random", "y"].mean()),
            "x_mean_by_cell": cl.groupby("cell")["x"].mean().round(4).to_dict()}


def rehearsal(seeds, d) -> dict:
    """B-T: Test IR을 가짜 OOS로. 시드 단위 R3·R5 반폭, S-b 보정 회귀 계수 반폭."""
    test, ir = d["test_ir"].to_numpy(), d["ir"].to_numpy()
    med = lambda s, a: float(np.median(test[s["cells"][a]["adopted"]])) if s["cells"][a]["adopted"] else np.nan
    out = {}
    for name, (a, b) in {"R3": MAIN_R1, "R5": ("lightgbm_ext_mmr05_resid", "random")}.items():
        dl = np.array([med(s, a) - med(s, b) for s in seeds])
        dl = dl[np.isfinite(dl)]
        lo, hi = boot_ci(dl, 0.95, f"rehearsal_{name}")
        out[name] = {"mean_diff": float(dl.mean()), "ci_95": [lo, hi], "halfwidth": (hi - lo) / 2,
                     "pseudo_verdict": "비열등" if lo > -MARGIN else ("열등" if hi < -MARGIN else "판정 불가")}
    for cell, (res, raw) in R2_CELLS.items():
        def coef(idx):
            rows = []
            for arm, dmy in ((raw, 0.0), (res, 1.0)):
                ids = sorted({i for k in idx for i in seeds[k]["cells"][arm]["adopted"]})
                rows += [(ir[i], dmy, test[i]) for i in ids]
            x = np.array(rows)
            X = np.column_stack([np.ones(len(x)), x[:, 0], x[:, 1]])
            return float(np.linalg.lstsq(X, x[:, 2], rcond=None)[0][2])
        rng = rng_for("kr_B_analysis", f"rehearsal_Sb_{cell}")
        boots = [coef(rng.integers(0, len(seeds), len(seeds))) for _ in range(2000)]
        lo, hi = np.quantile(boots, [0.0125, 0.9875])
        out[f"S-b_{cell}"] = {"coef": coef(range(len(seeds))), "ci_975": [float(lo), float(hi)],
                              "halfwidth": float((hi - lo) / 2)}
    out["note"] = "Test는 기준 ②로 채택에 쓰였으므로 편향이 있다. 단계 B 설정은 바꾸지 않는다 (설계서 11절)"
    return out


def run() -> dict:
    seeds = load_seeds()
    d = pd.read_parquet(R.rdir() / "directed.parquet")
    store = R._Store(np.load(R.rdir() / "is_excess_main.npy", mmap_mode="r"), d["direction"].to_numpy())
    cnt = count_table(seeds)
    cl = clusters(seeds, store)
    cl.to_parquet(SB.out_dir() / "clusters_is.parquet", index=False)
    out = {"n_seeds": len(seeds), "thresholds": json.loads((SB.out_dir() / "thresholds_main.json")
                                                           .read_text(encoding="utf-8"))["thresholds"],
           "count_means": cnt.mean().round(2).to_dict(), "count": count_tests(cnt), "S_a": s_a(seeds, d["ir"].to_numpy()),
           "D1_stageB": d1(cl), "rehearsal_test_as_oos": rehearsal(seeds, d),
           "k_refs": pd.Series([len(s["refs"]) for s in seeds]).value_counts().to_dict()}
    (SB.out_dir() / "analysis_is_test.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str),
                                                        encoding="utf-8")
    return out


def main() -> None:
    out = run()
    c = out["count"]
    for k in ("R1", "R2_base_topb", "R2_ext_mmr", "R4"):
        print(k, round(c[k]["mean_a"], 1), "vs", round(c[k]["mean_b"], 1), "차이", round(c[k]["mean_diff"], 2),
              [round(x, 2) for x in c[k]["ci"]], c[k]["verdict"])
    print("R2 묶음:", c["R2"]["verdict"], "| S-a:", {k: v["verdict"] for k, v in out["S_a"].items()})


if __name__ == "__main__":
    main()
