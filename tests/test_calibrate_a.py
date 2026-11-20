"""단계 A 조정: 방향, 탐욕 중복 제거, 임계값 접두사 성질 (krsas.calibrate_a). 합성 데이터."""
import numpy as np
import pandas as pd

from krsas import calibrate_a as C


def _store(series, direction):
    arr = np.stack([series, -series]).astype(np.float32)     # 0 = +, 1 = − (− 방향 초과수익은 부호만 다르게 둔다)
    return C.ExcessStore(arr, np.asarray(direction))


def test_directed_picks_larger_ir():
    res = pd.DataFrame({"formula_id": [1, 2], "monthly_ir_pos": [0.5, -0.2], "monthly_ir_neg": [0.1, 0.3],
                        "monthly_turnover_pos": [0.1, 0.2], "monthly_turnover_neg": [0.3, 0.4],
                        "monthly_te_ann_pos": [0, 0], "monthly_te_ann_neg": [0, 0],
                        "monthly_excess_ann_pos": [0, 0], "monthly_excess_ann_neg": [0, 0]})
    d = C.directed(res, "monthly")
    assert list(d["direction"]) == [1, -1] and list(d["ir"]) == [0.5, 0.3] and list(d["turnover"]) == [0.1, 0.4]


def test_greedy_drops_correlated_keeps_best():
    rng = np.random.default_rng(0)
    a = rng.normal(size=300)
    b = a + 0.1 * rng.normal(size=300)        # a와 상관 높음
    c = rng.normal(size=300)
    store = _store(np.stack([a, b, c]), [1, 1, 1])
    ir = np.array([0.5, 0.9, 0.3])
    kept = C.greedy_unique(np.array([0, 1, 2]), ir, store)
    assert list(kept) == [False, True, True]   # b(IR 높음)를 먼저 채택, a는 b와 상관 ≥ 0.7


def test_threshold_counts_monotone():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(50, 200))
    store = _store(x, np.ones(50, int))
    d = pd.DataFrame({"ir": rng.uniform(0, 1.6, 50), "turnover": rng.uniform(0, 1, 50)})
    counts = C.unique_by_threshold(np.arange(50), d, store, cap=0.9)
    assert (np.diff(counts) <= 0).all() and counts[0] <= 50
