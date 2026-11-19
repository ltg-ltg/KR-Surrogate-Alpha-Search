"""단계 A: `pool_v1` 전수 IS 전용 백테스트 (설계서 8.2절 리밸런싱 주기 결정, 기준 ③ 회전율, 마진 적정성).

    python -m krsas.stage_a --workers 16            # 남은 수식만 이어서 계산
    python -m krsas.stage_a --limit 200             # 시험

**IS만 본다.** 패널을 IS 마지막 거래일(2017-12-29 이전 마지막 거래일)까지 잘라서 계산하므로 Test·OOS 가격은
메모리에 올라오지도 않는다. 시그널은 주 유니버스로 한 번 계산하고, 주기 3개 × 방향 2개로 시뮬레이션한다.

산출 ({data_root}/stage_a/)
    results/part_*.parquet   formula_id, 필터 판정, 주기·방향별 IS 순 IR·초과수익·추적오차·회전율·비용, 시간
    excess/part_*.npz        주기별 float32 (2, n, IS 일수): 방향 +/− 의 IS 일별 순 초과수익 (기준 ④ 상관용)
    meta.json                IS 날짜, 벤치마크 지표, 설정
시그널 필터: Study 1 signal_filter(최소 유효 비율 0.70, 상수일 비율 ≤ 0.10)를 IS 기간·주 유니버스 구성 종목-일에 적용
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import data_root

IS = ("2011-01-01", "2017-12-31")
PERIODS = ("daily", "weekly", "monthly")
DIRS = (1.0, -1.0)
MIN_VALID, MAX_CONST = 0.70, 0.10      # Study 1 config/experiment.yaml pool.signal_filter
PART = 500
_W: dict = {}


SMOKE = {"on": False}


def out_dir() -> Path:
    return data_root() / ("stage_a_smoke" if SMOKE["on"] else "stage_a")


def _truncate_is(p):
    end = p.rows(*IS).stop
    return p.truncated(end)


def _init() -> None:
    import numba
    numba.set_num_threads(1)
    from . import backtest as B
    from . import panel as P
    p = _truncate_is(P.load())
    rows = p.rows(*IS)
    setups = {per: B.make_setup(p, "main", per) for per in PERIODS}
    bench = {per: B.benchmark(s) for per, s in setups.items()}
    _W.update(panel=p, rows=rows, setups=setups, bench=bench)


def _one(item):
    from . import signal as S          # Study 1 경로를 먼저 연다
    from . import backtest as B
    from src.generator.filters import signal_filter
    fid, formula = item
    p, rows = _W["panel"], _W["rows"]
    t0 = time.perf_counter()
    sig = S.signal(formula, p, p.member_main)
    t_sig = time.perf_counter() - t0
    fr = signal_filter(sig, p.member_main & p.valid, rows, MIN_VALID, MAX_CONST)
    row = {"formula_id": fid, "filter_pass": fr.passed, "filter_reason": fr.reason,
           "valid_ratio": fr.valid_ratio, "constant_ratio": fr.constant_day_ratio, "time_signal_sec": t_sig}
    ex = {}
    t0 = time.perf_counter()
    for per in PERIODS:
        st, bench = _W["setups"][per], _W["bench"][per]
        arr = np.empty((2, rows.stop - rows.start), dtype=np.float32)
        for k, d in enumerate(DIRS):
            s = B.portfolio(st, sig, d)
            m = B.metrics(s, bench, rows)
            name = "pos" if d > 0 else "neg"
            for key, v in m.items():
                row[f"{per}_{key}_{name}"] = v
            arr[k] = (s.net[rows] - bench.net[rows]).astype(np.float32)
        ex[per] = arr
    row["time_backtest_sec"] = time.perf_counter() - t0
    return row, ex


def run(workers: int, limit: int | None) -> None:
    from .pool import load_pool
    pool = load_pool()
    if limit:
        pool = pool.sample(limit, random_state=0)
        SMOKE["on"] = True
    items = list(zip(pool["formula_id"].astype(int), pool["formula"]))
    od = out_dir()
    (od / "results").mkdir(parents=True, exist_ok=True)
    (od / "excess").mkdir(parents=True, exist_ok=True)
    parts = [items[i:i + PART] for i in range(0, len(items), PART)]
    todo = [(k, part) for k, part in enumerate(parts) if not (od / "results" / f"part_{k:03d}.parquet").exists()]
    print(f"수식 {len(items)}개, 묶음 {len(parts)}개 중 남은 {len(todo)}개, 워커 {workers}", flush=True)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(workers, initializer=_init) as ex:
        for k, part in todo:
            res = list(ex.map(_one, part, chunksize=4))
            rows = [r for r, _ in res]
            np.savez(od / "excess" / f"part_{k:03d}.npz", formula_id=np.array([r["formula_id"] for r in rows]),
                     **{per: np.stack([e[per] for _, e in res]) for per in PERIODS})
            pd.DataFrame(rows).to_parquet(od / "results" / f"part_{k:03d}.parquet", index=False)
            done = todo.index((k, part)) + 1
            el = time.perf_counter() - t0
            print(f"  묶음 {k} 완료 ({done}/{len(todo)}), {el/60:.1f}분, 남은 예상 {el/done*(len(todo)-done)/60:.0f}분",
                  flush=True)
    _write_meta()


def _write_meta() -> None:
    from . import backtest as B
    from . import panel as P
    p = _truncate_is(P.load())
    rows = p.rows(*IS)
    meta = {"is": IS, "is_dates": [str(d.date()) for d in p.dates[rows]], "periods": PERIODS,
            "filter": {"min_valid_ratio": MIN_VALID, "max_constant_day_ratio": MAX_CONST}, "benchmark": {}}
    for per in PERIODS:
        st = B.make_setup(p, "main", per)
        b = B.benchmark(st)
        meta["benchmark"][per] = {"ann_net": float(np.mean(b.net[rows]) * B.ANN),
                                  "cost_ann": float(np.mean(b.cost[rows]) * B.ANN),
                                  "turnover": float(np.nanmean(b.turnover[rows])),
                                  "rebalances": int(st.reb[rows].sum())}
    (out_dir() / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


def load_results() -> pd.DataFrame:
    return pd.concat([pd.read_parquet(f) for f in sorted((out_dir() / "results").glob("part_*.parquet"))],
                     ignore_index=True)


def load_excess(period: str) -> tuple[np.ndarray, np.ndarray]:
    """(formula_id, (2, n, days) float32)."""
    ids, arrs = [], []
    for f in sorted((out_dir() / "excess").glob("part_*.npz")):
        z = np.load(f)
        ids.append(z["formula_id"])
        arrs.append(z[period])
    return np.concatenate(ids), np.concatenate(arrs, axis=0).transpose(1, 0, 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 6))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run(args.workers, args.limit)


if __name__ == "__main__":
    main()
