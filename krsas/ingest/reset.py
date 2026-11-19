"""기준가 재설정일 처리 (설계서 5.5절 처리 규칙, 단계 A 검증 2 결과로 구체화).

    python -m krsas.ingest.reset      # clean/returns.parquet에 r_adj, ratio_adj, adj_factor_adj, reset_kind를 더한다

순서: clean → adjust(기본 공식) → dart(crno 대응표) → rights.build_clean(권리일정 표) → reset

재개일(resume: 정지 다음 거래일, 기준가 ≠ 직전 종가)마다 그날의 기준가 비율 ratio = prev_close / base를
그날의 액면가·주식수 변화, 권리일정 사건 유형과 맞춰 아래로 나눈다. prev_close는 정지 중 조정일(halt_adjust)을
거친 값이라, 정지 중 거래소가 일반 주주 기준으로 반영한 감자·병합은 이미 들어 있다 (예: 차등감자).

    reset_kind   판정                                             r_adj (재개일)
    par          ratio ≈ 직전 액면가 / 액면가 (±1%)                r_basic (기준가가 분할·병합을 반영)
    shares       ratio ≈ 주식수 / 직전 주식수 (±1%)                r_basic
    spinoff      권리일정에 회사분할·분할합병 → config/reset_events.csv의 처리 (표에 없으면 멈춘다)
    reduction    권리일정에 자본감소·액면병합 → 표에 있으면 표의 처리, 없으면 r_basic (reviewed = False)
                   검토 안 된 감자에 주식수 비율을 쓰지 않는 이유: 차등감자(소액주주 비율 ≠ 전체 주식수 비율),
                   같은 정지 구간의 출자전환·유상소각 때문에 주식수 비율이 주주 비율과 크게 다를 수 있다
                   (2026-11-19 검토: 동양건설 2012 +951%, 코데즈컴바인 2015 +4,003% 같은 값이 나왔다).
                   KRX 기준가는 일반 주주 기준 감자 비율을 반영하고 재상장 시초가 결정분(0.5~2배)만 빠지므로 오차가 유계다
    표의 처리 (treatment)
                   total_return   (Σ 배정 주식수 × 그날 종가 + 현금) / prev_close − 1
                                  components 예: "self:auto;123450:auto", "self:0.3333333", "self:auto;cash:4945"
                                  self:auto = 주식수 / 직전 주식수, 종목:auto = 그 종목 주식수 / 분할 전 주식수,
                                  cash = 구주 1주당 현금(원, 유상감자 등)
                   base_fallback  r_basic (신설회사가 늦게 상장 등, 미래 가격을 쓰지 않는다)
                   reset          close / prev_close − 1
    reset        액면·주식수 변화와 권리일정 사건 모두 없음         close / prev_close − 1  (k = 1)
    other        그 밖 (증자만 있음 등)                             r_basic, 건수를 보고한다

수정계수: ratio_adj = prev_close × (1 + r_adj) / close 를 앞에서부터 누적한 adj_factor_adj (adj_factor와 같은 방식,
6.1절 "t까지 알려진 기업 행위만"). 재개일 밖의 날은 ratio_adj = ratio, r_adj = r_basic.
total_return은 그날 상장해 거래된 종목의 종가만 쓴다 (없으면 멈춘다). 그래서 미래 가격이 들어가지 않는다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import ROOT
from .clean import clean_dir

EVENTS = ROOT / "config" / "reset_events.csv"
TOL = 0.01
WINDOW_DAYS = 90            # 정지 시작 전 이만큼까지의 권리일정 사건을 본다
SPIN_ACTIONS = {"회사분할", "분할합병"}
REDUCTION_ACTIONS = {"자본감소", "액면병합"}
DATE_COLS = ["bas_dt", "ex_date", "list_date", "issue_date", "deliver_date", "merge_date"]


def load_events(path: Path = EVENTS) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")
    df["event_date"] = pd.to_datetime(df["event_date"])
    bad = (set(df["treatment"]) - {"total_return", "base_fallback", "reset"}) | \
        (set(df["kind"]) - {"spinoff", "reduction"})
    if bad:
        raise ValueError(f"reset_events.csv 값 오류: {bad}")
    if df.duplicated(["ticker", "event_date"]).any():
        raise ValueError("reset_events.csv에 같은 (종목, 날짜)가 두 번 있습니다")
    return df


def parse_components(spec: str) -> list[tuple[str, str]]:
    """'self:auto;123450:0.25' → [('self', 'auto'), ('123450', '0.25')]"""
    return [tuple(p.split(":", 1)) for p in spec.split(";") if p]


def classify(returns: pd.DataFrame, prices: pd.DataFrame, share_info: pd.DataFrame,
             rights: pd.DataFrame) -> pd.DataFrame:
    """재개일 행마다 reset_kind, sh_k, par_k, acts."""
    df = returns.merge(prices[["date", "ticker", "volume", "shares"]], on=["date", "ticker"])
    df = df.merge(share_info[["date", "ticker", "parval"]], on=["date", "ticker"])
    df = df.sort_values(["ticker", "date"], kind="stable").reset_index(drop=True)
    g = df.groupby("ticker")
    df["sh_prev"] = g["shares"].shift(1)
    df["par_prev"] = g["parval"].shift(1)
    traded = df["date"].where(df["volume"] > 0)
    df["last_traded"] = traded.groupby(df["ticker"]).shift(1)
    df["last_traded"] = df["last_traded"].groupby(df["ticker"]).ffill()
    ev = df[df["resume"]].copy()
    ev["sh_k"] = ev["shares"].astype(float) / ev["sh_prev"].astype(float)
    ev["par_k"] = ev["par_prev"].astype(float) / ev["parval"].astype(float)

    r = rights.dropna(subset=["ticker"]).copy()
    r["d_min"] = r[DATE_COLS].min(axis=1)
    r["d_max"] = r[DATE_COLS].max(axis=1)
    j = ev[["ticker", "date", "last_traded"]].merge(r[["ticker", "action", "d_min", "d_max"]], on="ticker")
    j = j[(j["d_max"] >= j["last_traded"] - pd.Timedelta(days=WINDOW_DAYS)) & (j["d_min"] <= j["date"])]
    acts = j.groupby(["ticker", "date"])["action"].agg(lambda s: set(s)).rename("acts")
    ev = ev.merge(acts, on=["ticker", "date"], how="left")
    ev["acts"] = ev["acts"].apply(lambda a: a if isinstance(a, set) else set())

    lr = np.log(ev["ratio"].astype(float))
    par_ok = ev["par_k"].notna() & ((ev["par_k"] - 1).abs() > 1e-9) & (np.abs(lr - np.log(ev["par_k"])) < TOL)
    sh_ok = ((ev["sh_k"] - 1).abs() > 1e-9) & (np.abs(lr - np.log(ev["sh_k"])) < TOL)
    spin = ev["acts"].apply(lambda a: bool(a & SPIN_ACTIONS))
    red = ev["acts"].apply(lambda a: bool(a & REDUCTION_ACTIONS))
    no_change = ((ev["sh_k"] - 1).abs() <= 1e-9) & (ev["par_k"].fillna(1).sub(1).abs() <= 1e-9) & \
        ev["acts"].apply(lambda a: not a)
    ev["reset_kind"] = np.select([par_ok, sh_ok, spin, red, no_change],
                                 ["par", "shares", "spinoff", "reduction", "reset"], "other")
    return ev[["date", "ticker", "close", "prev_close", "ratio", "r_basic", "sh_prev", "sh_k", "par_k", "acts",
               "reset_kind"]]


def adjusted_returns(ev: pd.DataFrame, prices: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """재개일의 r_adj, treatment(표에서 온 처리), reviewed."""
    px = prices.set_index(["date", "ticker"])
    close = px["close"].astype(float)
    shares_on = px["shares"].astype(float)
    table = events.set_index(["ticker", "event_date"])
    out = ev.copy()
    r_adj, treat, reviewed = [], [], []
    for e in out.itertuples():
        kind, P, P0 = e.reset_kind, float(e.close), float(e.prev_close)
        key = (e.ticker, e.date)
        t, ok = "", True
        if kind == "spinoff" and key not in table.index:
            raise KeyError(f"reset_events.csv에 없는 분할 재개일: {e.ticker} {e.date.date()}")
        if kind in ("spinoff", "reduction") and key in table.index:
            row = table.loc[key]
            if row["kind"] != kind:
                raise ValueError(f"{e.ticker} {e.date.date()}: 표의 kind {row['kind']} != 판정 {kind}")
            t = row["treatment"]
            if t == "total_return":
                v = 0.0
                for comp, n in parse_components(row["components"]):
                    if comp == "cash":
                        v += float(n)
                        continue
                    tk = e.ticker if comp == "self" else comp
                    if (e.date, tk) not in close.index:
                        raise KeyError(f"{e.ticker} {e.date.date()}: 배정 종목 {tk}의 그날 종가가 없습니다")
                    if n == "auto":         # 존속: 주식수 비율, 신설: 신설 주식수 / 분할 전 주식수 (지분율대로 배정)
                        shares = e.sh_k if comp == "self" else shares_on[(e.date, tk)] / float(e.sh_prev)
                    else:
                        shares = float(n)
                    v += shares * close[(e.date, tk)]
                r_adj.append(v / P0 - 1)
            elif t == "reset":
                r_adj.append(P / P0 - 1)
            else:
                r_adj.append(e.r_basic)
        elif kind == "reduction":
            r_adj.append(e.r_basic)
            ok = False
        elif kind == "reset":
            r_adj.append(P / P0 - 1)
        else:
            r_adj.append(e.r_basic)
        treat.append(t)
        reviewed.append(ok)
    out["r_adj"] = r_adj
    out["treatment"] = treat
    out["reviewed"] = reviewed
    return out


def build(returns: pd.DataFrame | None = None, prices: pd.DataFrame | None = None,
          share_info: pd.DataFrame | None = None, rights: pd.DataFrame | None = None,
          events: pd.DataFrame | None = None, write: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = clean_dir()
    base_cols = ["date", "ticker", "close", "base", "prev_close", "ratio", "r_basic", "adj_factor",
                 "listing", "resume", "halt_adjust", "gap"]
    returns = pd.read_parquet(root / "returns.parquet", columns=base_cols) if returns is None else returns[base_cols]
    prices = pd.read_parquet(root / "prices", columns=["date", "ticker", "close", "volume", "shares"]) \
        if prices is None else prices
    share_info = pd.read_parquet(root / "share_info", columns=["date", "ticker", "parval"]) \
        if share_info is None else share_info
    rights = pd.read_parquet(root / "rights.parquet") if rights is None else rights
    events = load_events() if events is None else events
    ev = adjusted_returns(classify(returns, prices, share_info, rights), prices, events)

    out = returns.merge(ev[["date", "ticker", "reset_kind", "treatment", "reviewed", "r_adj"]],
                        on=["date", "ticker"], how="left")
    out["reset_kind"] = out["reset_kind"].fillna("")
    out["treatment"] = out["treatment"].fillna("")
    out["reviewed"] = out["reviewed"].fillna(True).astype(bool)
    out["r_adj"] = out["r_adj"].where(out["reset_kind"] != "", out["r_basic"])
    out["ratio_adj"] = np.where(out["reset_kind"] != "",
                                out["prev_close"] * (1 + out["r_adj"]) / out["close"].astype(float), out["ratio"])
    out = out.sort_values(["ticker", "date"], kind="stable")
    out["adj_factor_adj"] = out.groupby("ticker")["ratio_adj"].cumprod()
    out = out.sort_values(["date", "ticker"]).reset_index(drop=True)
    if write:
        out.to_parquet(root / "returns.parquet", index=False)
    return out, ev


def main() -> None:
    out, ev = build()
    print("재개일", len(ev), ev["reset_kind"].value_counts().to_dict(),
          "/ 표 처리", ev["treatment"].replace("", "-").value_counts().to_dict(),
          "/ 검토 안 된 감자", int((~ev["reviewed"]).sum()))
    d = (ev["r_adj"] - ev["r_basic"]).abs()
    print("r_adj와 r_basic이 다른 재개일", int((d > 1e-12).sum()), "/ |차이| 중앙값", round(float(d[d > 1e-12].median()), 4))


if __name__ == "__main__":
    main()
