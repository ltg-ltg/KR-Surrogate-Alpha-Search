"""기준가 기반 일별 수익률, 기업 이벤트, 기준가 재설정일 후보 (설계서 5.5절).

    python -m krsas.ingest.adjust

kr_quant_machine `src/ingest/krx_adjust.py`(커밋 351ec2c)의 `price_ratios`·`classify`를 가져왔고,
5.5절의 재설정일 후보 표시를 더했다.

기본 공식 (5.5절)
    기준가 B_t = close_t − change_t,  일별 수익률 r_t = change_t / B_t = close_t / B_t − 1
    B_t ≠ 직전 거래일 종가인 날이 기업 이벤트다 (분할·병합·권리락·감자는 거래소가 기준가를 조정한다)
    수정계수 adj_factor는 앞에서부터 누적한다 (첫날 1, 이벤트일에 × prev_close / B_t).
    뒤에 생긴 이벤트가 과거 계수를 바꾸지 않으므로 6.1절 "t까지 알려진 기업 행위만 반영" 규칙을 지킨다.
    현금배당은 기준가에 반영되지 않는다 → 가격 수익률 (설계서 5절 "배당 반영하지 않음")

재설정일 후보 (5.5절 탐지 규칙 중 지금 데이터로 판정할 수 있는 부분)
    listing    종목이 처음 나타난 날. 수익률 계산에서 제외한다 (r_basic = NaN)
    resume     직전 거래일에 거래가 없었고(volume 0) 오늘 B_t ≠ 직전 종가인 날
    gap        직전 거래일에 그 종목이 응답에 없었고 오늘 B_t ≠ 직전 등장일 종가인 날
    ※ ① "권리일정정보에 같은 날 기업 행위가 없는 날" 판정과 처리 규칙 r_t = P_t / (P_{t-1} k_t) − 1은
      공공데이터포털 주식권리일정정보를 받은 뒤 넣는다. 그때까지 r_basic을 주 결과에 쓰지 않는다.

산출물
    {data_root}/clean/returns.parquet       date, ticker, close, base, prev_close, ratio, r_basic, adj_factor, listing, resume, gap
    {data_root}/clean/corp_actions.parquet  date, ticker, name, prev_close, base, ratio, kind, resume, gap,
                                            parval_prev, parval, shares_prev, shares
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .clean import clean_dir


def price_ratios(prices: pd.DataFrame) -> pd.DataFrame:
    """종목·날짜별 prev_close, base, ratio(= prev_close / base), 재설정일 후보 표시.

    prices는 거래일마다 그날 응답에 있는 종목 행이다. 직전 '거래일'은 시장 전체 날짜 순서로 본다.
    """
    df = prices[["date", "ticker", "name", "close", "change", "volume"]].sort_values(["ticker", "date"], kind="stable")
    df = df.reset_index(drop=True)
    dates = pd.Index(sorted(df["date"].unique()))
    pos = pd.Series(np.arange(len(dates)), index=dates)
    close = df["close"].astype(float)
    g = df.groupby("ticker")
    df["prev_close"] = close.groupby(df["ticker"]).shift(1)
    prev_pos = g["date"].shift(1).map(pos)
    prev_vol = g["volume"].shift(1)
    df["base"] = close - df["change"].astype(float)
    ok = df["prev_close"].gt(0) & df["base"].gt(0) & close.gt(0)
    df["ratio"] = np.where(ok, df["prev_close"] / df["base"], 1.0)

    df["listing"] = df["prev_close"].isna()
    shifted = ok & (df["ratio"] != 1.0)
    df["gap"] = shifted & (df["date"].map(pos) - prev_pos > 1)
    df["resume"] = shifted & ~df["gap"] & prev_vol.eq(0)
    df["r_basic"] = np.where(df["base"].gt(0) & ~df["listing"], close / df["base"] - 1, np.nan)
    return df


def classify(row) -> str:
    """액면가·주식수 변화로 이벤트 종류를 추정한다. 참고용이고 수익률 계산에는 쓰지 않는다."""
    par_changed = pd.notna(row.parval) and pd.notna(row.parval_prev) and row.parval != row.parval_prev
    if par_changed:
        return "split" if row.ratio > 1 else "reverse_split"
    if row.ratio > 1:
        return "ex_rights"            # 유·무상증자 권리락, 주식배당락 등 (기준가 하락)
    return "capital_reduction_or_merge"   # 감자·병합 뒤 거래 재개 등 (기준가 상승)


def build(prices: pd.DataFrame | None = None, share_info: pd.DataFrame | None = None,
          write: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = clean_dir()
    prices = pd.read_parquet(root / "prices") if prices is None else prices
    share_info = pd.read_parquet(root / "share_info") if share_info is None else share_info
    df = price_ratios(prices)
    df["adj_factor"] = df.groupby("ticker")["ratio"].cumprod()
    returns = df[["date", "ticker", "close", "base", "prev_close", "ratio", "r_basic", "adj_factor",
                  "listing", "resume", "gap"]].sort_values(["date", "ticker"]).reset_index(drop=True)

    events = df.loc[df["ratio"] != 1.0, ["date", "ticker", "name", "prev_close", "base", "ratio", "resume", "gap"]].copy()
    info = share_info[["date", "ticker", "parval", "shares"]].sort_values(["ticker", "date"], kind="stable")
    info = info.assign(parval=info["parval"].astype(float), shares=info["shares"].astype(float))
    info["parval_prev"] = info.groupby("ticker")["parval"].shift(1)
    info["shares_prev"] = info.groupby("ticker")["shares"].shift(1)
    events = events.merge(info, on=["date", "ticker"], how="left")
    events["kind"] = [classify(r) for r in events.itertuples()] if len(events) else []
    events = events[["date", "ticker", "name", "prev_close", "base", "ratio", "kind", "resume", "gap",
                     "parval_prev", "parval", "shares_prev", "shares"]].sort_values(["date", "ticker"])
    events = events.reset_index(drop=True)
    if write:
        root.mkdir(parents=True, exist_ok=True)
        returns.to_parquet(root / "returns.parquet", index=False)
        events.to_parquet(root / "corp_actions.parquet", index=False)
    return returns, events


def main() -> None:
    returns, events = build()
    print("수익률 행", len(returns), "/ 이벤트", len(events), events["kind"].value_counts().to_dict(),
          "/ 재설정일 후보 resume", int(returns["resume"].sum()), "gap", int(returns["gap"].sum()),
          "/ 신규 상장", int(returns["listing"].sum()))


if __name__ == "__main__":
    main()
