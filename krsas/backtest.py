"""롱온리 백테스트 (설계서 8.1~8.4절).

    t일 종가 시그널 → 유니버스 구성 종목 안에서 순위 → 방향(+ 상위 / − 하위) q = 20% 동일 가중 목표
    → t+1일 종가 체결 (비용도 t+1일) → 수익은 t+1 → t+2부터. 리밸런싱 사이에는 비중이 가격 따라 움직인다.

체결 규칙 (8.1절)
    - 체결일 거래정지 종목은 사지 않는다 (그 비중은 현금). 보유 중 거래정지 종목은 팔지 않고, 팔아야 하면
      재개일 종가에 판다 (재개일 수익률은 5.5절 r_adj)
    - 정지 종목을 못 팔아 현금이 모자라면 매수량을 같은 비율로 줄인다
    - 상장폐지: 마지막 거래일 종가에 청산 (매도 비용 전부, 슬리피지는 하위 구간 값, 5.2·8.3절). 이후 현금
    - 유니버스 재구성일(eff)에는 주기와 상관없이 리밸런싱한다 (편출 매도)
비용 (8.3절): 매도 = 거래세(체결일·시장) + 수수료 + 슬리피지, 매수 = 수수료 + 슬리피지.
    슬리피지 구간은 직전 재구성에서 배정한 구간 (편출 뒤에도 그 구간). 민감도 유니버스는 단일 값
    세 성분(거래세·수수료·슬리피지)을 따로 저장해 비용 민감도를 다시 계산할 수 있게 한다 (8.6절)
일별 순수익 = (1 + 총수익) × (1 − 비용) − 1. 비용은 그날 체결 전 가치 대비 비율이다
벤치마크 (8.4절): 같은 유니버스 동일 가중, 같은 주기·체결·비용 규칙
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numba import njit

from . import costs
from .panel import Panel

Q = 0.20
PERIODS = ("daily", "weekly", "monthly")
ANN = 252


# ---------------------------------------------------------------- 준비물 (유니버스·주기별로 한 번)
@dataclass
class Setup:
    panel: Panel
    universe: str
    member: np.ndarray          # (T, N) 시그널일 구성
    reb: np.ndarray             # (T,) 시그널일이면 True
    sell_tax: np.ndarray        # (T, N)
    sell_rate: np.ndarray       # (T, N) 거래세 + 수수료 + 슬리피지
    buy_rate: np.ndarray
    slip: np.ndarray            # (T, N) 슬리피지만
    delist_slip: float
    commission: float


def rebalance_rows(dates: pd.DatetimeIndex, period: str, member: np.ndarray) -> np.ndarray:
    """주기의 첫 거래일 + 재구성일. 구성 종목이 있는 날만."""
    s = pd.Series(np.arange(len(dates)), index=dates)
    if period == "daily":
        reb = np.ones(len(dates), dtype=bool)
    else:
        key = dates.to_period("W-SUN" if period == "weekly" else "M")
        reb = np.zeros(len(dates), dtype=bool)
        reb[s.groupby(key).min().to_numpy()] = True
    any_member = member.any(axis=1)
    eff = any_member & ~np.r_[False, any_member[:-1]] | np.r_[False, (member[1:] != member[:-1]).any(axis=1)]
    return (reb | eff) & any_member


def make_setup(panel: Panel, universe: str = "main", period: str = "monthly") -> Setup:
    cfg = costs.load()
    T, N = panel.member_main.shape
    member = panel.member_main if universe == "main" else panel.member_sens
    tax = np.empty((T, N))
    for mkt, flag in (("kospi", False), ("kosdaq", True)):
        rate = np.array([costs.sell_tax(mkt, d) for d in panel.dates])
        tax = np.where(panel.kosdaq == flag, rate[:, None], tax)
    if universe == "main":
        b = pd.DataFrame(np.where(panel.bucket_main >= 0, panel.bucket_main, np.nan)).ffill().to_numpy()
        table = cfg["slippage"]["main_universe"]
        lvl = np.array([table["top"], table["middle"], table["bottom"]])
        slip = np.where(np.isfinite(b), lvl[np.nan_to_num(b, nan=2).astype(int)], table["bottom"])
        delist = table[cfg["slippage"]["delisting_liquidation"]]
    else:
        slip = np.full((T, N), cfg["slippage"]["sensitivity_universe"])
        delist = cfg["slippage"]["sensitivity_universe"]
    comm = cfg["commission"]
    return Setup(panel, universe, member, rebalance_rows(panel.dates, period, member), tax,
                 tax + comm + slip, comm + slip, slip, float(delist), float(comm))


# ---------------------------------------------------------------- 목표 종목
@njit(cache=True)
def select_rows(sig, member, reb, direction, q):
    """시그널일마다 구성 종목 중 시그널이 있는 종목의 방향 상위 q. 동률은 열 순서."""
    T, N = sig.shape
    out = np.zeros((T, N), dtype=np.bool_)
    for t in range(T):
        if not reb[t]:
            continue
        idx = np.empty(N, dtype=np.int64)
        val = np.empty(N)
        n = 0
        for j in range(N):
            if member[t, j] and np.isfinite(sig[t, j]):
                idx[n] = j
                val[n] = -direction * sig[t, j]
                n += 1
        if n == 0:
            continue
        k = max(1, int(np.floor(q * n + 0.5)))
        order = np.argsort(val[:n], kind="mergesort")
        for r in range(k):
            out[t, idx[order[r]]] = True
    return out


# ---------------------------------------------------------------- 시뮬레이션
@njit(cache=True)
def simulate(target, reb, ret, tradable, last_row, sell_tax, slip, comm, delist_slip):
    """target: (T, N) 시그널일의 목표 종목. 반환: gross, cost_tax, cost_comm, cost_slip, turnover (T,)."""
    T, N = ret.shape
    w = np.zeros(N)
    cash = 1.0
    pending = np.zeros(N, dtype=np.bool_)
    gross = np.zeros(T)
    c_tax = np.zeros(T)
    c_comm = np.zeros(T)
    c_slip = np.zeros(T)
    turn = np.full(T, np.nan)
    for d in range(T):
        # 1 그날 수익 (시작 비중)
        g = 0.0
        for j in range(N):
            if w[j] != 0.0:
                r = ret[d, j]
                if np.isfinite(r):
                    g += w[j] * r
                    w[j] *= 1.0 + r
        tot = cash
        for j in range(N):
            tot += w[j]
        gross[d] = g
        if tot > 0:
            cash /= tot
            for j in range(N):
                w[j] /= tot
        tax = 0.0
        com = 0.0
        sl = 0.0
        traded = 0.0
        # 2 상장폐지 청산
        for j in range(N):
            if w[j] != 0.0 and d == last_row[j] and last_row[j] < T - 1:
                a = w[j]
                c = a * (sell_tax[d, j] + comm + delist_slip)
                tax += a * sell_tax[d, j]
                com += a * comm
                sl += a * delist_slip
                cash += a - c
                w[j] = 0.0
                pending[j] = False
        # 3 밀린 매도 (재개일)
        for j in range(N):
            if pending[j] and tradable[d, j] and w[j] != 0.0:
                a = w[j]
                tax += a * sell_tax[d, j]
                com += a * comm
                sl += a * slip[d, j]
                cash += a - a * (sell_tax[d, j] + comm + slip[d, j])
                traded += a
                w[j] = 0.0
                pending[j] = False
        # 4 리밸런싱 체결 (전날이 시그널일)
        if d > 0 and reb[d - 1]:
            k = 0
            for j in range(N):
                if target[d - 1, j]:
                    k += 1
            tw = 1.0 / k if k > 0 else 0.0
            for j in range(N):           # 매도 먼저
                if w[j] == 0.0:
                    pending[j] = False
                    continue
                want = tw if target[d - 1, j] else 0.0
                if w[j] > want:
                    if tradable[d, j]:
                        a = w[j] - want
                        tax += a * sell_tax[d, j]
                        com += a * comm
                        sl += a * slip[d, j]
                        cash += a - a * (sell_tax[d, j] + comm + slip[d, j])
                        traded += a
                        w[j] = want
                        pending[j] = False
                    elif want == 0.0:
                        pending[j] = True
                else:
                    pending[j] = False
            # 매수: 매수액 + 비용 ≤ 현금. 모자라면 같은 비율로 줄인다
            need = 0.0
            for j in range(N):
                if target[d - 1, j] and tradable[d, j] and w[j] < tw:
                    need += (tw - w[j]) * (1.0 + comm + slip[d, j])
            fill = 1.0 if need <= cash else (max(cash, 0.0) / need if need > 0 else 0.0)
            for j in range(N):
                if target[d - 1, j] and tradable[d, j] and w[j] < tw:
                    a = (tw - w[j]) * fill
                    com += a * comm
                    sl += a * slip[d, j]
                    cash -= a * (1.0 + comm + slip[d, j])
                    traded += a
                    w[j] += a
            turn[d] = traded / 2.0
        cost = tax + com + sl
        c_tax[d] = tax
        c_comm[d] = com
        c_slip[d] = sl
        if cost > 0:                      # 비용은 체결 때 현금에서 빠졌다. 전체를 다시 1로
            scale = 1.0 / (1.0 - cost)
            cash *= scale
            for j in range(N):
                w[j] *= scale
    return gross, c_tax, c_comm, c_slip, turn


@dataclass
class Series:
    gross: np.ndarray
    cost_tax: np.ndarray
    cost_comm: np.ndarray
    cost_slip: np.ndarray
    turnover: np.ndarray

    @property
    def cost(self) -> np.ndarray:
        return self.cost_tax + self.cost_comm + self.cost_slip

    @property
    def net(self) -> np.ndarray:
        return (1.0 + self.gross) * (1.0 - self.cost) - 1.0


def run(setup: Setup, target: np.ndarray) -> Series:
    p = setup.panel
    out = simulate(target, setup.reb, p.ret, p.tradable, p.last_row, setup.sell_tax, setup.slip,
                   setup.commission, setup.delist_slip)
    return Series(*out)


def portfolio(setup: Setup, sig: np.ndarray, direction: float) -> Series:
    return run(setup, select_rows(sig, setup.member, setup.reb, direction, Q))


def benchmark(setup: Setup) -> Series:
    return run(setup, setup.member & setup.reb[:, None])


# ---------------------------------------------------------------- 지표
def ir(excess: np.ndarray) -> float:
    x = excess[np.isfinite(excess)]
    if x.size < 2:
        return 0.0
    sd = x.std(ddof=1)
    return float(x.mean() / sd * np.sqrt(ANN)) if sd > 0 else 0.0


def metrics(s: Series, bench: Series, rows: slice) -> dict:
    ex = s.net[rows] - bench.net[rows]
    t = s.turnover[rows]
    return {"ir": ir(ex), "excess_ann": float(np.nanmean(ex) * ANN),
            "te_ann": float(np.nanstd(ex, ddof=1) * np.sqrt(ANN)),
            "turnover": float(np.nanmean(t)) if np.isfinite(t).any() else 0.0,
            "cost_ann": float(np.nanmean(s.cost[rows]) * ANN)}
