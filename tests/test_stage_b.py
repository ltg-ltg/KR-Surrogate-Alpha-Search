"""단계 B 도우미: 통과 기준 ①~③, 사전 필터, 잔차 대표 채우기 (합성 데이터)."""
import numpy as np
import pandas as pd

from krsas import calibrate_b as CB
from krsas import replay_b as R


def _d(ir, test, to, sk=None):
    n = len(ir)
    return pd.DataFrame({"ir": ir, "test_ir": test, "turnover": to, "direction": np.ones(n, int),
                         "skeleton": sk if sk is not None else [f"s{i}" for i in range(n)]})


def _store(x):
    arr = np.stack([x, -x]).astype(np.float32)
    return R._Store(arr, np.ones(x.shape[0], int))


def test_pass_1to3_uses_all_three():
    d = _d([0.6, 0.6, 0.6, 0.4], [0.5, 0.1, 0.5, 0.5], [0.5, 0.5, 0.9, 0.5])
    assert list(CB.pass_1to3(d, np.arange(4), 0.5, 0.3)) == [True, False, False, False]


def test_prefilter_skips_skeleton_of_passed_signal():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(6, 200))
    d = _d([0.9, 0.9, 0.2, 0.9, 0.2, 0.2], [0.5] * 6, [0.1] * 6, sk=["a", "a", "b", "c", "c", "d"])
    old_b = R.B
    R.B = 4
    try:
        picks = R.prefilter_picks(np.arange(6), d, _store(x), {"is_ir": 0.5, "test_ir": 0.3})
    finally:
        R.B = old_b
    assert picks == [0, 2, 3, 5]          # 1은 a(통과), 4는 c(통과)라 건너뛰고 예산 밖


def test_residual_refs_fill_drops_only_criterion_1():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(8, 300))
    ir = np.array([0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    to = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.95, 0.1])     # 6은 회전율 초과 → 채우기에서도 빠진다
    y, refs = R.residual_targets(np.arange(8), ir, to, _store(x), 0.5, 0.792)
    assert refs[0] == 0 and len(refs) == 5 and 6 not in refs
    assert y.shape == (8,)
