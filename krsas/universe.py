"""시점별 유니버스 (설계서 5.1절, 5.4절 제외 규칙, 6.1절 누수 방지).

    python -m krsas.universe              # {data_root}/clean/universe.parquet, universe_stats.parquet

재구성: 매년 6월·12월 첫 거래일(eff)에 바꾸고, 그 직전 거래일(base)까지의 정보만 쓴다.
첫 재구성은 2010-12 (IS 2011-01 시작 전에 유니버스가 있어야 한다. 2010년은 lookback 버퍼).
`[단계 A 확정 필요]` 설계서 5.1절은 "매년 6월·12월"만 정한다. 첫 거래일 적용·직전 거래일 정보는 이 모듈의 선택이다.

제외 규칙 (5.4절, 단계 A 데이터 검증 7·9 결과로 정한 형태)
    1 종목기본정보가 날짜별 스냅샷이므로(검증 7) stock_type == 보통주 & secugrp == 주권만 남긴다.
      우선주·종류주권(영문 혼합 코드 27개 포함, 검증 9), 외국주권, 부동산투자회사(리츠), 선박투자회사,
      투자회사, 사회간접자본투융자회사(인프라펀드), 주식예탁증권(DR)이 여기서 빠진다
    2 끝자리 코드 규칙: 1에서 모두 판정되므로 쓰지 않는다 (1 통과 종목의 끝자리는 모두 0, 검증 9)
    3 종목명 규칙: 스팩만 남는다. base일 **그날의 종목명**에 "스팩|SPAC"이 있거나 소속부가 SPAC인 종목.
      스팩은 합병 뒤 같은 코드로 사업회사가 되므로 과거·미래 이름을 쓰지 않는다.
      설계서의 리츠·인프라 이름 패턴은 쓰지 않는다: 1이 이미 걸러내고, 이름 패턴은 메리츠화재·두산인프라코어 같은
      보통주를 잘못 뺀다

유동성 필터 (주 유니버스)
    - 거래대금: 제외 규칙 뒤 전 종목에서 base일까지 20거래일 평균 거래대금(value)의 하위 20% 제외.
      상장 20거래일 미만이라 평균이 없는 종목도 제외한다 `[단계 A 확정 필요: 설계서에 없음]`
    - 거래정지: base일 volume == 0
    - 관리종목: base일 소속부 == "관리종목(소속부없음)". 소속부는 코스닥 2011-05-02부터만 있고 코스피는 전 기간 빈 값이다
      (검증 8). 설계서 5.1절 규칙대로 소속부가 없는 시장·날짜에는 관리종목 조건을 빼고 거래대금·거래정지만 쓴다
    - 남은 종목 중 시가총액(mktcap = 상장주식수 × 종가) 상위 500
민감도 유니버스: 제외 규칙 뒤 코스피 종목 중 base일 거래정지가 아닌 시가총액 상위 200
슬리피지 구간 (8.3절): 주 유니버스 안에서 20일 평균 거래대금 3분위 (top / middle / bottom)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .ingest.clean import clean_dir

MAIN_SIZE = 500
SENS_SIZE = 200
LIQ_WINDOW = 20
LIQ_CUT = 0.20
FIRST_EFF_MONTH = "2010-12"
ADMIN_SECT = "관리종목(소속부없음)"
SPAC_SECT = "SPAC(소속부없음)"
SPAC_NAME = r"스팩|SPAC"


def rebalance_dates(dates) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """(eff, base) 목록. eff = 6월·12월 첫 거래일, base = 그 직전 거래일."""
    dates = pd.DatetimeIndex(sorted(pd.unique(pd.Series(dates))))
    out = []
    for i in range(1, len(dates)):
        d = dates[i]
        if d.month in (6, 12) and dates[i - 1].month != d.month and d >= pd.Timestamp(FIRST_EFF_MONTH):
            out.append((d, dates[i - 1]))
    return out


def add_liquidity(prices: pd.DataFrame) -> pd.DataFrame:
    """종목별 base일까지 20거래일 평균 거래대금. 거래정지일(value 0)도 하루로 센다.
    응답에서 빠진 날이 없으므로(검증 1) 종목의 행 20개 = 20거래일이다."""
    df = prices.sort_values(["ticker", "date"], kind="stable").copy()
    df["val20"] = (df.groupby("ticker")["value"].transform(
        lambda s: s.astype(float).rolling(LIQ_WINDOW, min_periods=LIQ_WINDOW).mean()))
    return df


def exclusion(day: pd.DataFrame) -> pd.DataFrame:
    """제외 규칙 판정 열을 붙인다 (rule1_out, spac_out, eligible)."""
    day = day.copy()
    day["rule1_out"] = ~((day["stock_type"] == "보통주") & (day["secugrp"] == "주권"))
    name = day["name"].fillna("")
    day["spac_out"] = ~day["rule1_out"] & (name.str.contains(SPAC_NAME, regex=True) | (day["sect"] == SPAC_SECT))
    day["eligible"] = ~day["rule1_out"] & ~day["spac_out"]
    return day


def select(day: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """base일 하루치(prices + share_info + val20)에서 주·민감도 유니버스를 고른다."""
    day = exclusion(day)
    pool = day[day["eligible"]].copy()
    thr = pool["val20"].quantile(LIQ_CUT)
    sect_known = {m: (pool.loc[pool["market"] == m, "sect"].fillna("") != "").any() for m in pool["market"].unique()}
    pool["liq_out"] = pool["val20"].isna() | (pool["val20"] < thr)
    pool["halt_out"] = pool["volume"] == 0
    pool["admin_out"] = (pool["sect"] == ADMIN_SECT) & pool["market"].map(sect_known)
    keep = pool[~(pool["liq_out"] | pool["halt_out"] | pool["admin_out"])]
    main = keep.nlargest(MAIN_SIZE, "mktcap").copy()
    main["bucket"] = pd.qcut(main["val20"].rank(method="first", ascending=False), 3,
                             labels=["top", "middle", "bottom"]).astype(str)
    sens = pool[(pool["market"] == "KOSPI") & ~pool["halt_out"]].nlargest(SENS_SIZE, "mktcap")
    stats = {
        "rows": len(day), "rule1_out": int(day["rule1_out"].sum()), "spac_out": int(day["spac_out"].sum()),
        "eligible": len(pool), "liq_threshold": float(thr), "val20_missing": int(pool["val20"].isna().sum()),
        "liq_out": int(pool["liq_out"].sum()), "halt_out": int(pool["halt_out"].sum()),
        "admin_out": int(pool["admin_out"].sum()),
        "admin_known_kospi": bool(sect_known.get("KOSPI", False)),
        "admin_known_kosdaq": bool(sect_known.get("KOSDAQ", False)),
        "remain": len(keep), "main": len(main), "main_kosdaq": int((main["market"] == "KOSDAQ").sum()),
        "sens": len(sens), "main_min_mktcap": float(main["mktcap"].min()) if len(main) else np.nan,
    }
    main = main.assign(universe="main")[["ticker", "market", "mktcap", "val20", "bucket", "universe"]]
    sens = sens.assign(universe="sens", bucket=None)[["ticker", "market", "mktcap", "val20", "bucket", "universe"]]
    return pd.concat([main, sens], ignore_index=True), stats


def build(prices: pd.DataFrame | None = None, share_info: pd.DataFrame | None = None,
          write: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = clean_dir()
    if prices is None:
        prices = pd.read_parquet(root / "prices", columns=["date", "ticker", "name", "market", "sect",
                                                           "close", "volume", "value", "mktcap"])
    if share_info is None:
        share_info = pd.read_parquet(root / "share_info", columns=["date", "ticker", "stock_type", "secugrp"])
    df = add_liquidity(prices.merge(share_info[["date", "ticker", "stock_type", "secugrp"]], on=["date", "ticker"]))
    by_date = dict(tuple(df.groupby("date")))
    members, stats = [], []
    for eff, base in rebalance_dates(df["date"]):
        m, s = select(by_date[base])
        members.append(m.assign(eff=eff, base=base))
        stats.append({"eff": eff, "base": base, **s})
    members = pd.concat(members, ignore_index=True)[["eff", "base", "universe", "ticker", "market", "mktcap",
                                                     "val20", "bucket"]]
    stats = pd.DataFrame(stats)
    if write:
        members.to_parquet(root / "universe.parquet", index=False)
        stats.to_parquet(root / "universe_stats.parquet", index=False)
    return members, stats


def holdings(members: pd.DataFrame, last_date: pd.Timestamp, universe: str = "main") -> pd.DataFrame:
    """종목별 편입 구간 [eff, end). end는 다음 재구성일 (마지막 구간은 last_date 다음 날)."""
    m = members[members["universe"] == universe]
    effs = sorted(m["eff"].unique())
    ends = dict(zip(effs, effs[1:] + [pd.Timestamp(last_date) + pd.Timedelta(days=1)]))
    return m.assign(end=m["eff"].map(ends))[["eff", "end", "ticker"]]


def main() -> None:
    members, stats = build()
    print(stats[["eff", "eligible", "liq_out", "halt_out", "admin_out", "remain", "main", "main_kosdaq", "sens"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
