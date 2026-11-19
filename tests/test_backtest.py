"""시그널(시점 기준 척도·유니버스 횡단면)과 롱온리 시뮬레이션. 작은 합성 패널.

설계서 5.3절 검증 11(절단 테스트)·12(비용·체결 손 계산)의 단위 테스트 판이다. 실제 데이터 판은 krsas.validation.
"""
import numpy as np
import pandas as pd
import pytest

from krsas import backtest as B
from krsas import signal as S
from krsas.panel import FIELDS, Panel


def make_panel(close, volume=None, scale=None, member=None, kosdaq=None, last_row=None, ret=None):
    close = np.asarray(close, dtype=float)
    T, N = close.shape
    volume = np.full((T, N), 100.0) if volume is None else np.asarray(volume, float)
    scale = np.ones((T, N)) if scale is None else np.asarray(scale, float)
    valid = volume > 0
    f = {"open": close.copy(), "high": close * 1.01, "low": close * 0.99, "close": close.copy(),
         "volume": volume / scale}
    for k in ("open", "high", "low", "close"):
        f[k] = np.where(valid, f[k] * scale, np.nan)
    f["volume"] = np.where(valid, f["volume"], np.nan)
    r = np.full((T, N), np.nan)
    r[1:] = close[1:] / close[:-1] - 1
    if ret is not None:
        r = np.asarray(ret, float)
    f["returns"] = np.where(valid, r, np.nan)
    return Panel(pd.bdate_range("2020-01-01", periods=T), np.array([f"T{j}" for j in range(N)]), f, scale, valid,
                 np.nan_to_num(r), valid.copy(), np.full(N, T - 1) if last_row is None else np.asarray(last_row),
                 np.zeros((T, N), bool) if kosdaq is None else kosdaq,
                 np.ones((T, N), bool) if member is None else member, np.ones((T, N), bool),
                 np.zeros((T, N), np.int8))


# ---------------------------------------------------------------- 시그널
def test_exponents():
    assert S.exponents(S.parse("mul(close, volume)")) == (1, 1)
    assert S.exponents(S.parse("div(ts_mean(close, 5), volume)")) == (1, -1)
    assert S.exponents(S.parse("rank(close)")) == (0, 0)


def test_rank_uses_actual_price_after_split():
    # A는 둘째 날 10대 1 분할: 원주가 1000 → 100, 앞쪽 수정계수 10. B는 실제 가격 500
    close = [[1000, 500], [100, 500]]
    scale = [[1, 1], [10, 1]]
    p = make_panel(close, scale=scale)
    sig = S.signal("close", p, np.ones((2, 2), bool))
    assert sig[1, 0] == pytest.approx(100) and sig[1, 1] == pytest.approx(500)       # 실제 척도
    r = S.evaluate(S.parse("rank(close)"), p, np.ones((2, 2), bool))
    assert r[1, 0] < r[1, 1]                                                          # A(100) < B(500)
    # 시계열 연산은 수정 척도로: 분할을 건너는 변화율은 0
    d = S.signal("div(close, ts_delay(close, 1))", p, np.ones((2, 2), bool))
    assert d[1, 0] == pytest.approx(1.0)


def test_cross_section_only_members():
    close = [[1, 2, 3], [1, 2, 3]]
    member = np.array([[True, True, False], [True, True, True]])
    p = make_panel(close, member=member)
    r = S.evaluate(S.parse("rank(close)"), p, member)
    assert np.isnan(r[0, 2]) and r[0, 1] == pytest.approx(1.0) and r[1, 2] == pytest.approx(1.0)


@pytest.mark.parametrize("formula", [
    "ts_mean(close, 5)", "rank(ts_delta(close, 3))", "ts_corr(rank(volume), returns, 10)",
    "zscore(div(close, ts_max(high, 5)))", "ts_rank(mul(close, volume), 5)",
])
def test_truncation_no_lookahead(formula):
    rng = np.random.default_rng(0)
    T, N, t = 40, 6, 25
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, (T, N)), axis=0))
    scale = np.ones((T, N))
    scale[30:, 2] = 5.0                       # t 뒤의 분할
    member = rng.random((T, N)) > 0.2
    full = make_panel(close, scale=scale, member=member)
    s_full = S.signal(formula, full, member)
    s_cut = S.signal(formula, full.truncated(t + 1), member[: t + 1])
    np.testing.assert_allclose(s_full[: t + 1], s_cut, equal_nan=True)
    alt = close.copy()
    alt[t + 1:] *= rng.uniform(0.5, 2.0, (T - t - 1, N))
    s_alt = S.signal(formula, make_panel(alt, scale=scale, member=member), member)
    np.testing.assert_allclose(s_full[: t + 1], s_alt[: t + 1], equal_nan=True)


# ---------------------------------------------------------------- 시뮬레이션 (손 계산)
def _sim(ret, target, reb, tradable=None, last_row=None, tax=0.003, slip=0.001, comm=0.00015, delist_slip=0.002):
    ret = np.asarray(ret, float)
    T, N = ret.shape
    tradable = np.ones((T, N), bool) if tradable is None else np.asarray(tradable)
    last_row = np.full(N, T - 1) if last_row is None else np.asarray(last_row)
    return B.Series(*B.simulate(np.asarray(target), np.asarray(reb), ret, tradable, last_row,
                                np.full((T, N), tax) if np.isscalar(tax) else np.asarray(tax),
                                np.full((T, N), slip), comm, delist_slip))


def test_execution_lag_and_costs_by_hand():
    # 시그널일 0 → 체결일 1 종가 (A, B 반씩) → 수익은 2일부터
    ret = [[0.0, 0.0], [0.5, 0.5], [0.10, -0.10], [0.0, 0.0]]
    target = [[True, True], [False, False], [False, False], [False, False]]
    reb = [True, False, False, False]
    s = _sim(ret, target, reb)
    assert s.gross[1] == 0.0                                   # 체결일 수익은 이전 보유(현금)
    buy = 1.0 / (1 + 0.00115) * 0.00115                         # 매수액 a: a × (1 + 비용률) = 현금 1
    assert s.cost[1] == pytest.approx(buy) and s.cost_tax[1] == 0.0
    assert s.gross[2] == pytest.approx(0.5 * 0.10 + 0.5 * -0.10)
    assert s.turnover[1] == pytest.approx(0.5 / (1 + 0.00115))


def test_rebalance_sell_tax_and_drift():
    # 0일 시그널 A만 → 1일 매수. 2일 A +100%. 2일 시그널 B만 → 3일 A 전량 매도, B 매수
    ret = [[0, 0], [0, 0], [1.0, 0], [0, 0], [0, 0.2]]
    target = np.zeros((5, 2), bool)
    target[0, 0] = True
    target[2, 1] = True
    reb = [True, False, True, False, False]
    tax = np.full((5, 2), 0.003)
    tax[3:] = 0.0025                                           # 세율 변경 경계: 체결일 세율
    s = _sim(ret, target, reb, tax=tax)
    v1 = 1 - 0.00115 / 1.00115                                 # 매수 비용 뒤 가치
    assert s.net[1] == pytest.approx(v1 - 1)
    sell = 1.0 * (0.0025 + 0.00015 + 0.001)                   # 3일: 체결 전 가치 대비 비중 1 매도
    buy = (1.0 - sell) / (1 + 0.00115) * 0.00115               # 남은 현금으로 매수액 + 비용
    assert s.cost_tax[3] == pytest.approx(0.0025)
    assert s.cost[3] == pytest.approx(sell + buy)
    assert s.gross[4] == pytest.approx(0.2)


def test_halted_holding_sold_on_resume_and_halted_buy_stays_cash():
    # 0일 시그널 A·B → 1일 B는 정지라 못 산다 (B 몫 현금). 2일 시그널 없음(A) → 3일 A 정지라 못 판다 → 4일 재개 때 판다
    ret = [[0, 0], [0, 0], [0, 0], [0, 0], [-0.5, 0], [0, 0]]
    tradable = np.ones((6, 2), bool)
    tradable[1, 1] = False
    tradable[3, 0] = False
    target = np.zeros((6, 2), bool)
    target[0] = True
    reb = [True, False, True, False, False, False]
    s = _sim(ret, target, reb, tradable=tradable, tax=0.0, slip=0.0, comm=0.0)
    assert s.turnover[1] == pytest.approx(0.25)               # A만 0.5 매수
    assert s.turnover[3] == pytest.approx(0.0)                # A 정지: 매도 밀림
    assert s.gross[4] == pytest.approx(0.5 * -0.5)            # 재개일 수익은 보유자가 받는다
    s2 = _sim(ret, target, reb, tradable=tradable, tax=0.01, slip=0.0, comm=0.0)
    assert s2.cost_tax[4] > 0                                 # 재개일 종가에 매도


def test_delisting_liquidation_bottom_slippage():
    ret = [[0, 0], [0, 0], [0.1, 0], [np.nan, 0], [np.nan, 0]]
    target = np.zeros((5, 2), bool)
    target[0, 0] = True
    reb = [True, False, False, False, False]
    s = _sim(ret, target, reb, last_row=[2, 4], tax=0.003, slip=0.0005, comm=0.00015, delist_slip=0.002)
    assert s.cost[2] == pytest.approx(1.0 * (0.003 + 0.00015 + 0.002))
    assert s.gross[3] == 0.0 and s.gross[4] == 0.0            # 이후 현금


def test_select_rows_direction_and_q():
    sig = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, np.nan]])
    member = np.ones((1, 6), bool)
    reb = np.array([True])
    top = B.select_rows(sig, member, reb, 1.0, 0.4)
    bot = B.select_rows(sig, member, reb, -1.0, 0.4)
    assert list(np.where(top[0])[0]) == [3, 4] and list(np.where(bot[0])[0]) == [0, 1]
