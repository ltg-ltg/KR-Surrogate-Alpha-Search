"""단계 B-1: `pool_v1` 4만 전수 백테스트, 전 구간 (사전 등록 docs/preregistration.md 4절).

    python -m krsas.stage_b --workers 6         # 남은 묶음만 이어서 계산
    python -m krsas.stage_b --limit 32          # 시험 (stage_b_smoke/, OOS도 따로)

평가 변형 (선택은 주 설정, 평가만 바꾼다. 사전 등록 5절)
    main         주 유니버스, 월간 (주 결과)
    main_slip5   슬리피지 모든 구간 +5bp (상장폐지 청산 포함)
    main_slip15  +15bp
    main_weekly  주간 리밸런싱
    main_rbasic  보유 수익률을 기본 공식 r_basic으로 (재설정일 처리 영향, 보조)
    sens         민감도 유니버스 (코스피 상위 200), 월간. 시그널도 민감도 유니버스 횡단면으로 다시 계산
    코스피200 벤치마크와 상장폐지 대안 규칙은 이 모듈에 없다 (지수·정리매매 시작일 자료가 따로 필요하다)

산출 ({data_root}/stage_b/)
    is_test/part_*.parquet           수식별 IS·Test 지표 (변형 × 방향), 시그널 필터, 시간
    is_excess/<변형>/part_*.npy      float32 (2, n, IS 일수) 방향 +/− IS 일별 순 초과수익 (기준 ④)
    test_excess/main/part_*.npy      float32 (2, n, Test 일수) 주 결과 Test 일별 순 초과수익
    oos_locked/                      **OOS 지표와 일별 초과수익. krsas.oos만 읽는다** (설계서 13절, tests/test_oos_lock.py)
    meta.json                        구간 날짜, 벤치마크 지표, 변형 정의
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import data_root

PERIODS = {"is": ("2011-01-01", "2017-12-31"), "test": ("2018-01-01", "2019-12-31"),
           "oos": ("2020-01-01", "2023-12-31")}
VARIANTS = ("main", "main_slip5", "main_slip15", "main_weekly", "main_rbasic", "sens")
DIRS = (1.0, -1.0)
MIN_VALID, MAX_CONST = 0.70, 0.10
PART = 500
SMOKE = {"on": False}
_W: dict = {}


def out_dir() -> Path:
    return data_root() / ("stage_b_smoke" if SMOKE["on"] else "stage_b")


def oos_dir() -> Path:
    """OOS 잠금 폴더. 쓰기는 이 모듈, 읽기는 krsas.oos만 한다."""
    return out_dir() / "oos_locked"


def ensure_ret_basic() -> Path:
    from . import panel as P
    from .ingest.clean import clean_dir
    path = P.npy_dir() / "ret_basic.npy"
    if not path.exists():
        q = P.load_mmap()
        r = pd.read_parquet(clean_dir() / "returns.parquet", columns=["date", "ticker", "r_basic"],
                            filters=[("ticker", "in", list(q.tickers))])
        a = r.pivot(index="date", columns="ticker", values="r_basic").reindex(index=q.dates, columns=q.tickers) \
            .to_numpy(dtype=np.float64)
        np.save(path, np.where(np.isfinite(np.asarray(q.ret)), np.nan_to_num(a, nan=0.0), np.nan))
    return path


def _setups(p):
    from . import backtest as B
    base = B.make_setup(p, "main", "monthly")
    out = {"main": (base, p.ret), "main_weekly": (B.make_setup(p, "main", "weekly"), p.ret),
           "main_rbasic": (base, np.load(ensure_ret_basic(), mmap_mode="r")),
           "sens": (B.make_setup(p, "sens", "monthly"), p.ret)}
    for name, add in (("main_slip5", 0.0005), ("main_slip15", 0.0015)):
        out[name] = (replace(base, slip=base.slip + add, sell_rate=base.sell_rate + add, buy_rate=base.buy_rate + add,
                             delist_slip=base.delist_slip + add), p.ret)
    return out


def _sim(st, target, ret):
    from . import backtest as B
    p = st.panel
    return B.Series(*B.simulate(target, st.reb, ret, p.tradable, p.last_row, st.sell_tax, st.slip, st.commission,
                                st.delist_slip))


def _init() -> None:
    import numba
    numba.set_num_threads(1)
    from . import backtest as B
    from . import panel as P
    p = P.load_mmap()
    rows = {k: p.rows(*v) for k, v in PERIODS.items()}
    setups = _setups(p)
    bench = {k: _sim(st, st.member & st.reb[:, None], ret) for k, (st, ret) in setups.items()}
    _W.update(panel=p, rows=rows, setups=setups, bench=bench, B=B)


def _one(item):
    from . import signal as S
    from src.generator.filters import signal_filter
    B = _W["B"]
    fid, formula = item
    p, rows = _W["panel"], _W["rows"]
    t0 = time.perf_counter()
    sig = {"main": S.signal(formula, p, p.member_main), "sens": S.signal(formula, p, p.member_sens)}
    t_sig = time.perf_counter() - t0
    fr = signal_filter(sig["main"], p.member_main & p.valid, rows["is"], MIN_VALID, MAX_CONST)
    row = {"formula_id": fid, "filter_pass": fr.passed, "filter_reason": fr.reason, "valid_ratio": fr.valid_ratio,
           "constant_ratio": fr.constant_day_ratio, "time_signal_sec": t_sig}
    oos = {"formula_id": fid}
    ex_is, ex_test, ex_oos = {}, None, {}
    t0 = time.perf_counter()
    for var, (st, ret) in _W["setups"].items():
        s_ = sig["sens"] if var == "sens" else sig["main"]
        bench = _W["bench"][var]
        a_is = np.empty((2, rows["is"].stop - rows["is"].start), np.float32)
        a_test = np.empty((2, rows["test"].stop - rows["test"].start), np.float32)
        a_oos = np.empty((2, rows["oos"].stop - rows["oos"].start), np.float32)
        for k, d in enumerate(DIRS):
            s = _sim(st, B.select_rows(s_, st.member, st.reb, d, B.Q), ret)
            name = "pos" if d > 0 else "neg"
            for per in ("is", "test"):
                for key, v in B.metrics(s, bench, rows[per]).items():
                    row[f"{var}_{per}_{key}_{name}"] = v
            for key, v in B.metrics(s, bench, rows["oos"]).items():
                oos[f"{var}_oos_{key}_{name}"] = v
            ex = s.net - bench.net
            a_is[k], a_test[k], a_oos[k] = ex[rows["is"]], ex[rows["test"]], ex[rows["oos"]]
        ex_is[var] = a_is
        if var == "main":
            ex_test = a_test
        if var in ("main", "sens"):
            ex_oos[var] = a_oos
    row["time_backtest_sec"] = time.perf_counter() - t0
    return row, oos, ex_is, ex_test, ex_oos


def run(workers: int, limit: int | None) -> None:
    from .pool import load_pool
    pool = load_pool()
    if limit:
        pool = pool.sample(limit, random_state=0)
        SMOKE["on"] = True
    ensure_ret_basic()
    items = list(zip(pool["formula_id"].astype(int), pool["formula"]))
    od, ld = out_dir(), oos_dir()
    for sub in ["is_test", "test_excess/main", *[f"is_excess/{v}" for v in VARIANTS]]:
        (od / sub).mkdir(parents=True, exist_ok=True)
    for sub in ["metrics", "excess_main", "excess_sens"]:
        (ld / sub).mkdir(parents=True, exist_ok=True)
    parts = [items[i:i + PART] for i in range(0, len(items), PART)]
    todo = [k for k in range(len(parts)) if not (od / "is_test" / f"part_{k:03d}.parquet").exists()]
    print(f"수식 {len(items)}개, 묶음 {len(parts)}개 중 남은 {len(todo)}개, 작업자 {workers}", flush=True)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(workers, initializer=_init) as ex:
        for n, k in enumerate(todo, start=1):
            res = list(ex.map(_one, parts[k], chunksize=4))
            tag = f"part_{k:03d}"
            for v in VARIANTS:
                np.save(od / "is_excess" / v / f"{tag}.npy", np.stack([r[2][v] for r in res], axis=1))
            np.save(od / "test_excess" / "main" / f"{tag}.npy", np.stack([r[3] for r in res], axis=1))
            for v in ("main", "sens"):
                np.save(ld / f"excess_{v}" / f"{tag}.npy", np.stack([r[4][v] for r in res], axis=1))
            pd.DataFrame([r[1] for r in res]).to_parquet(ld / "metrics" / f"{tag}.parquet", index=False)
            pd.DataFrame([r[0] for r in res]).to_parquet(od / "is_test" / f"{tag}.parquet", index=False)  # 완료 표시
            el = time.perf_counter() - t0
            print(f"  묶음 {k} 완료 ({n}/{len(todo)}), {el / 60:.1f}분, 남은 예상 {el / n * (len(todo) - n) / 60:.0f}분",
                  flush=True)
    _write_meta()


def _write_meta() -> None:
    from . import panel as P
    p = P.load_mmap()
    setups = _setups(p)
    meta = {"periods": PERIODS, "variants": VARIANTS,
            "dates": {k: [str(d.date()) for d in p.dates[p.rows(*v)]] for k, v in PERIODS.items() if k != "oos"},
            "n_oos_days": int(p.rows(*PERIODS["oos"]).stop - p.rows(*PERIODS["oos"]).start),
            "benchmark_is_test": {}}
    from . import backtest as B
    for var, (st, ret) in setups.items():
        b = _sim(st, st.member & st.reb[:, None], ret)
        meta["benchmark_is_test"][var] = {per: {"ann_net": float(np.mean(b.net[p.rows(*PERIODS[per])]) * B.ANN),
                                                "cost_ann": float(np.mean(b.cost[p.rows(*PERIODS[per])]) * B.ANN)}
                                          for per in ("is", "test")}
    (out_dir() / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


# ---------------------------------------------------------------- 읽기 (IS·Test만)
def load_results() -> pd.DataFrame:
    return pd.concat([pd.read_parquet(f) for f in sorted((out_dir() / "is_test").glob("part_*.parquet"))],
                     ignore_index=True)


def load_is_excess(variant: str = "main") -> np.ndarray:
    """(2, n, IS 일수) float32, 풀 순서."""
    return np.concatenate([np.load(f) for f in sorted((out_dir() / "is_excess" / variant).glob("part_*.npy"))], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 2) - 2)))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run(args.workers, args.limit)


if __name__ == "__main__":
    main()
