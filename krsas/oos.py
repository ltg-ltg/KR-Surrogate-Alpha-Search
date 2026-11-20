"""단계 B-4: OOS 1회 (설계서 13절, 사전 등록 4절). **OOS 잠금 폴더를 읽는 유일한 모듈이다.**

    python -m krsas.oos --dry-run      # 시험: OOS 자리에 Test를 넣는다. stage_b/oos_dryrun/에 쓰고 잠그지 않는다
    python -m krsas.oos --run          # 실제 OOS 1회. 끝나면 잠금 파일을 만들고 다시 실행하지 않는다

실행 전 점검 (13절)
    - 레포에 커밋되지 않은 변경(추적 파일)이 있으면 거부한다. 실행 커밋 해시와 시도를 attempts.jsonl에 남긴다
    - 완료 잠금 파일(oos_locked/DONE.lock)이 있으면 거부한다
    - 재생성 대조 (사전 등록 4절): B-1 결과 중 모든 칸의 채택 시그널 합집합 + 무작위 2,000개 수식의 IS·Test 지표를
      다시 계산해 수식별 IR 차이 ≤ 1e-9인지 본다. 시드·칸별 고유 통과 집합을 B-1 결과로 다시 세어 리플레이와 같은지 본다.
      하나라도 어긋나면 분석 전에 멈춘다
판정 (설계서 10·11절)
    R3 (주 칸 − 무작위), R5 (확장·MMR·잔차 − 무작위): 시드마다 채택 시그널 OOS 순 IR 중앙값, 시드 대응 차이의
        부트스트랩 95% 구간. 비열등 = 하한 > −0.1, 열등 = 상한 < −0.1, 그 외 판정 불가
    S-b (칸마다): 보정 회귀 OOS IR ~ IS IR + 잔차 더미 (고유 시그널 단위, 시드 고정효과 없음), 시드 부트스트랩 97.5%.
        동등 = 구간이 (−0.1, +0.1) 안, 열등 = 상한 < −0.1, 우월 = 하한 > +0.1, 그 외 판정 불가.
        보조: IS 매칭 (빈 폭 0.1, 빈당 두 arm 모두 고유 10개 이상, 공통 구간 시그널 30% 미만이면 매칭 불가)
    보조 품질: OOS IR > 0 비율, OOS/IS IR 비 중앙값, 칸별 채택 시그널 동일 가중 결합 포트폴리오 OOS IR, 연도별 IR
    민감도 (12절): 변형별 R1·R2(IS·Test, 임계값 고정)와 R3. 민감도 유니버스는 다시 조정한 임계값 결과도 함께
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import analysis_b as AB
from . import calibrate_a as CA
from . import calibrate_b as CB
from . import replay_b as R
from . import stage_b as SB
from . import study1
from .paths import ROOT

study1.ensure()
from src.common.seeds import rng_for  # noqa: E402

MARGIN = 0.1
R3_CELLS = {"R3": ("lightgbm_base_topb_raw", "random"), "R5": ("lightgbm_ext_mmr05_resid", "random")}
SENS_VARIANTS = ("main_slip5", "main_slip15", "main_weekly", "main_rbasic", "sens")
TOL = 1e-9


# ---------------------------------------------------------------- 점검
def git_state() -> tuple[str, list[str]]:
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True, check=True).stdout.splitlines()
    return head, dirty


def out_dir(dry: bool) -> Path:
    return SB.out_dir() / "oos_dryrun" if dry else SB.oos_dir()


def log_attempt(dry: bool, **kw) -> None:
    d = out_dir(dry)
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "attempts.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"at": datetime.now().isoformat(timespec="seconds"), "dry_run": dry, **kw},
                            ensure_ascii=False) + "\n")


def regeneration_check(seeds, res) -> dict:
    """채택 합집합 + 무작위 2,000개의 IS·Test 지표를 다시 계산해 B-1과 맞춘다. 시드별 고유 통과를 다시 센다."""
    from concurrent.futures import ProcessPoolExecutor

    from .pool import load_pool
    pool = load_pool()
    adopted = sorted({i for s in seeds for c in s["cells"].values() for i in c["adopted"]})
    extra = rng_for("kr_B_oos_regen").choice(len(pool), 2000, replace=False)
    idx = sorted(set(adopted) | set(int(i) for i in extra))
    items = list(zip(pool["formula_id"].to_numpy()[idx].astype(int), pool["formula"].to_numpy()[idx]))
    with ProcessPoolExecutor(6, initializer=SB._init) as ex:
        rows = [r[0] for r in ex.map(SB._one, items, chunksize=8)]
    new = pd.DataFrame(rows).set_index("formula_id")
    old = res.set_index("formula_id").loc[new.index]
    cols = [c for c in new.columns if "_is_" in c or "_test_" in c]
    diff = float((new[cols] - old[cols]).abs().max().max())
    th = json.loads((SB.out_dir() / "thresholds_main.json").read_text(encoding="utf-8"))["thresholds"]
    d = CB.directed(res, "main")
    store = R._Store(np.load(R.rdir() / "is_excess_main.npy", mmap_mode="r"), d["direction"].to_numpy())
    bad = []
    for s in seeds:
        picks = np.load(R.rdir() / f"picks_{s['seed']:03d}.npz")
        for cell, v in s["cells"].items():
            n, ad = CB.count_unique(picks[cell].astype(np.int64), d, store, th["is_ir"], th["test_ir"])
            if n != v["unique"] or sorted(ad.tolist()) != sorted(v["adopted"]):
                bad.append((s["seed"], cell))
    return {"n_formulas_recomputed": len(idx), "max_abs_diff_is_test": diff, "pass_set_mismatch": len(bad),
            "ok": bool(diff <= TOL and not bad), "mismatch_examples": bad[:10]}


# ---------------------------------------------------------------- 읽기
def load_quality(dry: bool, variant: str = "main") -> pd.DataFrame:
    """formula_id, 방향 적용 IS IR, 평가 구간(OOS 또는 dry-run이면 Test) IR."""
    res = SB.load_results()
    d = CB.directed(res, variant)
    if dry:
        q = np.where(d["direction"] == 1, res[f"{variant}_test_ir_pos"], res[f"{variant}_test_ir_neg"])
    else:
        m = pd.concat([pd.read_parquet(f) for f in sorted((SB.oos_dir() / "metrics").glob("part_*.parquet"))],
                      ignore_index=True)
        if not np.array_equal(m["formula_id"].to_numpy(), res["formula_id"].to_numpy()):
            raise RuntimeError("OOS 지표 순서가 B-1 결과와 다릅니다")
        q = np.where(d["direction"] == 1, m[f"{variant}_oos_ir_pos"], m[f"{variant}_oos_ir_neg"])
    return pd.DataFrame({"formula_id": d["formula_id"], "is_ir": d["ir"], "q": q, "direction": d["direction"]})


def daily_eval(dry: bool) -> np.ndarray:
    """(2, n, 일수) 주 결과의 평가 구간 일별 순 초과수익."""
    if dry:
        return np.concatenate([np.load(f) for f in sorted((SB.out_dir() / "test_excess" / "main").glob("*.npy"))], axis=1)
    return np.concatenate([np.load(f) for f in sorted((SB.oos_dir() / "excess_main").glob("*.npy"))], axis=1)


# ---------------------------------------------------------------- 판정
def noninferiority(seeds, q, a, b, name) -> dict:
    med = lambda s, c: float(np.median(q[s["cells"][c]["adopted"]])) if s["cells"][c]["adopted"] else np.nan
    dl = np.array([med(s, a) - med(s, b) for s in seeds])
    dl = dl[np.isfinite(dl)]
    lo, hi = AB.boot_ci(dl, 0.95, f"oos_{name}")
    v = "비열등" if lo > -MARGIN else ("열등" if hi < -MARGIN else "판정 불가")
    return {"cell": a, "vs": b, "mean_diff": float(dl.mean()), "ci_95": [lo, hi], "verdict": v,
            "seeds_used": int(len(dl)), "median_a": float(np.nanmedian([med(s, a) for s in seeds])),
            "median_b": float(np.nanmedian([med(s, b) for s in seeds]))}


def _pooled_rows(seeds, idx, raw, res, is_ir, q):
    rows = []
    for arm, dmy in ((raw, 0.0), (res, 1.0)):
        ids = sorted({i for k in idx for i in seeds[k]["cells"][arm]["adopted"]})
        rows += [(is_ir[i], dmy, q[i]) for i in ids]
    return np.array(rows)


def s_b(seeds, is_ir, q) -> dict:
    out = {}
    for cell, (res, raw) in AB.R2_CELLS.items():
        def coef(idx):
            x = _pooled_rows(seeds, idx, raw, res, is_ir, q)
            X = np.column_stack([np.ones(len(x)), x[:, 0], x[:, 1]])
            return float(np.linalg.lstsq(X, x[:, 2], rcond=None)[0][2])
        rng = rng_for("kr_B_analysis", f"oos_Sb_{cell}")
        boots = [coef(rng.integers(0, len(seeds), len(seeds))) for _ in range(2000)]
        lo, hi = (float(v) for v in np.quantile(boots, [0.0125, 0.9875]))
        v = "동등" if (lo > -MARGIN and hi < MARGIN) else ("비동등 (잔차 열등)" if hi < -MARGIN else
                                                         ("비동등 (잔차 우월)" if lo > MARGIN else "판정 불가"))
        # 보조: IS 매칭
        x = _pooled_rows(seeds, range(len(seeds)), raw, res, is_ir, q)
        bins = np.floor(x[:, 0] / 0.1).astype(int)
        df = pd.DataFrame({"bin": bins, "arm": x[:, 1], "q": x[:, 2]})
        cnt = df.groupby(["bin", "arm"]).size().unstack(fill_value=0)
        common = cnt[(cnt.get(0.0, 0) >= 10) & (cnt.get(1.0, 0) >= 10)].index
        share = {int(a): float(df[(df["arm"] == a) & df["bin"].isin(common)].shape[0] / max((df["arm"] == a).sum(), 1))
                 for a in (0.0, 1.0)}
        matched = None
        if len(common) and min(share.values()) >= 0.3:
            g = df[df["bin"].isin(common)].groupby(["bin", "arm"])["q"].mean().unstack()
            w = cnt.loc[common].sum(axis=1)
            matched = float(((g[1.0] - g[0.0]) * w).sum() / w.sum())
        out[cell] = {"coef": coef(range(len(seeds))), "ci_975": [lo, hi], "verdict": v,
                     "matching": {"common_bins": [int(b) for b in common], "share_in_common": share,
                                  "matchable": matched is not None, "weighted_diff": matched}}
    return out


def s_interpretation(sa: dict, sb: dict) -> dict:
    """설계서 3절 S-a·S-b 해석표 (칸마다)."""
    out = {}
    for cell in AB.R2_CELLS:
        a, b = sa[cell]["verdict"], sb[cell]["verdict"]
        if a == "판정 불가":
            out[cell] = "판정 불가로 보고한다 (S-a 판정 불가)"
        elif a == "깨짐":
            out[cell] = "잔차 타깃이 IS IR이 낮은 시그널을 고르지 않았다. 선택 효과의 전제가 성립하지 않는다"
        elif b == "동등":
            out[cell] = "잔차 타깃의 품질 열세는 선택 효과다"
        elif b == "비동등 (잔차 열등)":
            out[cell] = "같은 IS 수준에서도 품질이 낮은 진짜 교환 관계다"
        elif b == "비동등 (잔차 우월)":
            out[cell] = "같은 IS에서는 잔차가 낫다. 원시 품질 열세는 선택 위치 때문이다"
        else:
            out[cell] = "선택 효과 여부를 판정할 수 없다"
    return out


def aux_quality(seeds, frame, ex, dates_per_year) -> dict:
    """보조: OOS IR > 0 비율, OOS/IS 비, 결합 포트폴리오 IR, 연도별 IR (칸별)."""
    q, is_ir, dirn = frame["q"].to_numpy(), frame["is_ir"].to_numpy(), frame["direction"].to_numpy()
    k = np.where(dirn == 1, 0, 1)
    cells = ["random", "lightgbm_base_topb_raw", "lightgbm_base_topb_resid", "lightgbm_ext_mmr05_raw",
             "lightgbm_ext_mmr05_resid"]
    out = {}
    for c in cells:
        pos, ratio, comb, yearly = [], [], [], []
        for s in seeds:
            ad = np.array(s["cells"][c]["adopted"], dtype=int)
            if not len(ad):
                continue
            pos.append(float((q[ad] > 0).mean()))
            ratio.append(float(np.median(q[ad] / is_ir[ad])))
            port = np.nanmean(np.stack([ex[k[i], i] for i in ad]).astype(np.float64), axis=0)
            comb.append(CA_ir(port))
            if dates_per_year:
                yearly.append({y: CA_ir(port[sl]) for y, sl in dates_per_year.items()})
        out[c] = {"share_q_pos_median": float(np.median(pos)), "q_over_is_median": float(np.median(ratio)),
                  "combined_ir_median": float(np.median(comb)),
                  "yearly_combined_ir_median": ({y: float(np.median([r[y] for r in yearly])) for y in yearly[0]}
                                                if yearly else None)}
    return out


def CA_ir(x: np.ndarray) -> float:
    from . import backtest as B
    return B.ir(x)


def sensitivity(seeds, dry: bool) -> dict:
    """변형별 R1·R2 (임계값 고정, IS·Test)와 R3 (평가 구간)."""
    res = SB.load_results()
    th = json.loads((SB.out_dir() / "thresholds_main.json").read_text(encoding="utf-8"))["thresholds"]
    out = {}
    variants = list(SENS_VARIANTS)
    for var in variants:
        d = CB.directed(res, var)
        store = R._Store(SB.load_is_excess(var), d["direction"].to_numpy())
        ths = {"fixed": th}
        if var == "sens":
            f = SB.out_dir() / "thresholds_sens.json"
            if not f.exists():
                CB.calibrate("sens")
            t2 = json.loads(f.read_text(encoding="utf-8"))
            if t2.get("status") == "ok":
                ths["recalibrated"] = t2["thresholds"]
        q = load_quality(dry, var)["q"].to_numpy()
        for tname, t in ths.items():
            vs = []
            for s in seeds:
                picks = np.load(R.rdir() / f"picks_{s['seed']:03d}.npz")
                cells = {}
                for c in ("random", "lightgbm_base_topb_raw", "lightgbm_base_topb_resid", "lightgbm_ext_mmr05_raw",
                          "lightgbm_ext_mmr05_resid"):
                    n, ad = CB.count_unique(picks[c].astype(np.int64), d, store, t["is_ir"], t["test_ir"])
                    cells[c] = {"unique": n, "adopted": ad.tolist()}
                vs.append({"seed": s["seed"], "cells": cells})
            cnt = pd.DataFrame([{c: v["unique"] for c, v in s_["cells"].items()} for s_ in vs])
            r = {}
            for name, (a, b, conf, exp) in {"R1": (*AB.MAIN_R1, 0.95, "neg"),
                                            "R2_base_topb": (*AB.R2_CELLS["base_topb"], 0.975, "pos"),
                                            "R2_ext_mmr": (*AB.R2_CELLS["ext_mmr"], 0.975, "pos")}.items():
                dl = (cnt[a] - cnt[b]).to_numpy(float)
                lo, hi = AB.boot_ci(dl, conf, f"sens_{var}_{tname}_{name}")
                r[name] = {"mean_diff": float(dl.mean()), "ci": [lo, hi], "verdict": AB.verdict_direction(lo, hi, exp)}
            r["R3"] = noninferiority(vs, q, *AB.MAIN_R1, f"sens_{var}_{tname}")
            out[f"{var}|{tname}"] = r
    return out


def run(dry: bool) -> dict:
    head, dirty = git_state()
    if dirty:
        raise SystemExit(f"커밋되지 않은 변경이 있어 실행하지 않습니다: {dirty[:5]}")
    od = out_dir(dry)
    lock = od / "DONE.lock"
    if not dry and lock.exists():
        raise SystemExit(f"OOS는 이미 실행했습니다 ({lock}). 다시 실행하지 않습니다")
    log_attempt(dry, commit=head, status="start")
    seeds = AB.load_seeds()
    res = SB.load_results()
    regen = regeneration_check(seeds, res)
    log_attempt(dry, commit=head, status="regeneration", **{k: v for k, v in regen.items() if k != "mismatch_examples"})
    if not regen["ok"]:
        log_attempt(dry, commit=head, status="stopped: regeneration mismatch")
        raise SystemExit(f"재생성 대조 실패: {regen}")
    frame = load_quality(dry)
    q, is_ir = frame["q"].to_numpy(), frame["is_ir"].to_numpy()
    meta = json.loads((SB.out_dir() / "meta.json").read_text(encoding="utf-8"))
    years = None
    if not dry:
        oos_dates = pd.bdate_range("2020-01-01", "2023-12-31")      # 연도 경계만 쓴다 (실제 거래일 수로 자른다)
        n = meta["n_oos_days"]
        from . import panel as P
        p = P.load_mmap()
        dts = p.dates[p.rows(*SB.PERIODS["oos"])]
        assert len(dts) == n
        years = {int(y): np.flatnonzero(dts.year == y) for y in sorted(set(dts.year))}
        del oos_dates
    sa = json.loads((SB.out_dir() / "analysis_is_test.json").read_text(encoding="utf-8"))["S_a"]
    out = {"dry_run": dry, "evaluation_period": "Test (가짜 OOS)" if dry else "OOS 2020~2023", "commit": head,
           "regeneration": regen}
    out |= {k: noninferiority(seeds, q, a, b, ("dry_" if dry else "") + k) for k, (a, b) in R3_CELLS.items()}
    out["S_b"] = s_b(seeds, is_ir, q)
    out["S_interpretation"] = s_interpretation(sa, out["S_b"])
    out["aux_quality"] = aux_quality(seeds, frame, daily_eval(dry), years)
    out["sensitivity"] = sensitivity(seeds, dry)
    od.mkdir(parents=True, exist_ok=True)
    (od / "results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    log_attempt(dry, commit=head, status="done")
    if not dry:
        lock.write_text(json.dumps({"commit": head, "at": datetime.now().isoformat(timespec="seconds")}),
                        encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--run", action="store_true")
    args = parser.parse_args()
    out = run(dry=args.dry_run)
    print(out["evaluation_period"], "| 재생성", out["regeneration"]["ok"])
    if args.dry_run:
        for k in R3_CELLS:
            print(k, out[k]["verdict"], [round(x, 3) for x in out[k]["ci_95"]])
        print("S-b", {c: v["verdict"] for c, v in out["S_b"].items()})


if __name__ == "__main__":
    main()
