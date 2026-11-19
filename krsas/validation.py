"""단계 A 데이터 검증 (설계서 5.3절, 체크리스트 3절).

    python -m krsas.validation        # {data_root}/validation/stageA_data.json + 상세 표 parquet

원자료 필드·건수만 본다. 수식·포트폴리오 성과는 계산하지 않는다 (OOS 구간 행도 건수 확인에는 들어간다).
상세 표(종목·날짜·가격)는 KRX 가공 시세라 데이터 폴더에만 쓰고, 레포에는 집계만 옮긴다.

항목                          함수
 1 상장폐지·일봉 완전성       check_delisting
 2 기준가 재설정일 (재개일)   check_resume
 3 유니버스 종목 수·필터      check_universe   (krsas.universe)
 4 첫 날짜·응답 필드          check_first_day
 7 종목기본정보 스냅샷        check_info_snapshot
 8 소속부로 관리종목 식별     check_sect
 9 종목코드 형식              check_ticker_format
10 전일대비 부호·무거래일     check_no_trade_days
 5(시세 대조)·6(키 기한)·11(절단)·12(손 계산)·13(검정력)은 여기서 하지 않는다
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import universe as uni
from .ingest.clean import clean_dir
from .ingest.krx import read_raw
from .paths import DATA_START, data_root

SERVICES = ("stk_trd", "ksq_trd", "stk_info", "ksq_info")
EXPECTED_TRD = {"BAS_DD", "ISU_CD", "ISU_NM", "MKT_NM", "SECT_TP_NM", "TDD_OPNPRC", "TDD_HGPRC", "TDD_LWPRC",
                "TDD_CLSPRC", "CMPPREVDD_PRC", "FLUC_RT", "ACC_TRDVOL", "ACC_TRDVAL", "MKTCAP", "LIST_SHRS"}
TOL = 0.01      # 수정 비율 일치 허용 (로그 차이)


def out_dir() -> Path:
    return data_root() / "validation"


def load() -> dict[str, pd.DataFrame]:
    root = clean_dir()
    prices = pd.read_parquet(root / "prices")
    info = pd.read_parquet(root / "share_info")
    returns = pd.read_parquet(root / "returns.parquet")
    df = prices.merge(info[["date", "ticker", "stock_type", "secugrp", "parval"]], on=["date", "ticker"])
    df = df.merge(returns[["date", "ticker", "base", "prev_close", "ratio", "listing", "resume", "halt_adjust",
                           "gap"]],
                  on=["date", "ticker"])
    df = df.sort_values(["ticker", "date"], kind="stable").reset_index(drop=True)
    df["common"] = (df["stock_type"] == "보통주") & (df["secugrp"] == "주권")
    return {"prices": prices, "info": info, "df": df}


# ---- 4 ----------------------------------------------------------------------------------------------------
def check_first_day() -> dict:
    out = {}
    for s in SERVICES:
        rows = read_raw(s, DATA_START)
        keys = set(rows[0]) if rows else set()
        out[s] = {"rows": len(rows), "fields": sorted(keys)}
    for s in ("stk_trd", "ksq_trd"):
        keys = set(out[s]["fields"])
        out[s]["missing_vs_design"] = sorted(EXPECTED_TRD - keys)
        out[s]["extra_vs_design"] = sorted(keys - EXPECTED_TRD)
    return out


# ---- 7 ----------------------------------------------------------------------------------------------------
def check_info_snapshot(prices: pd.DataFrame, info: pd.DataFrame, probe: str = "2012-06-01") -> dict:
    keys = prices[["date", "ticker"]].merge(info[["date", "ticker"]], how="outer", indicator=True)
    last = prices.groupby("ticker")["date"].max()
    day = info[info["date"] == probe]
    later_gone = day[day["ticker"].map(last) < pd.Timestamp(probe[:4] + "-12-31")]
    return {
        "key_match": keys["_merge"].value_counts().to_dict(),
        "probe_date": probe, "probe_rows": len(day), "probe_rows_delisted_same_year": len(later_gone),
        "tickers_with_name_change": int((info.groupby("ticker")["name"].nunique() > 1).sum()),
        "tickers_with_parval_change": int((info.groupby("ticker")["parval"].nunique() > 1).sum()),
        "verdict": "날짜별 스냅샷 (이후 상장폐지 종목 포함, 이름·액면가가 날짜마다 다름)",
    }


# ---- 8 ----------------------------------------------------------------------------------------------------
def check_sect(prices: pd.DataFrame) -> dict:
    s = prices.assign(has=prices["sect"].fillna("") != "", year=prices["date"].dt.year)
    by = s.groupby(["market", "year"])["has"].mean().unstack(0).round(3)
    kq = s[(s["market"] == "KOSDAQ") & s["has"]]
    first = kq["date"].min()
    after = s[(s["market"] == "KOSDAQ") & (s["date"] >= first)].groupby("date")["has"].mean()
    return {
        "nonempty_share_by_year": {str(k): v for k, v in by.to_dict().items()},
        "kosdaq_first_sect_date": str(first.date()) if pd.notna(first) else None,
        "kosdaq_dates_after_first_with_missing_sect": int((after < 1).sum()),
        "kosdaq_sect_values": kq["sect"].value_counts().to_dict(),
        "kospi_nonempty_rows": int(s.loc[s["market"] == "KOSPI", "has"].sum()),
        "admin_rows_kosdaq": int((prices["sect"] == uni.ADMIN_SECT).sum()),
        "verdict": "코스닥 2011-05-02부터 식별 가능, 코스피는 전 기간 불가 → 코스피와 코스닥 2010-12 재구성은 "
                   "관리종목 조건 없이 거래대금·거래정지만 (설계서 5.1절 대체 규칙)",
    }


# ---- 9 ----------------------------------------------------------------------------------------------------
def check_ticker_format(df: pd.DataFrame) -> dict:
    t = df.drop_duplicates("ticker", keep="last")
    alpha = t[~t["ticker"].str.fullmatch(r"\d{6}")]
    com = t[t["common"]]
    return {
        "tickers": len(t), "length_values": t["ticker"].str.len().value_counts().to_dict(),
        "alnum_tickers": len(alpha),
        "alnum_first_date": str(df[df["ticker"].isin(alpha["ticker"])]["date"].min().date()) if len(alpha) else None,
        "alnum_stock_type": alpha["stock_type"].value_counts().to_dict(),
        "alnum_common": int(alpha["common"].sum()),
        "common_tail_not_0": int((~com["ticker"].str.endswith("0")).sum()),
        "verdict": "영문 혼합 코드는 2010년부터 있으나 모두 우선주·종류주권 → 규칙 1(보통주·주권)로 빠진다. "
                   "규칙 1 통과 종목은 끝자리가 모두 0",
    }


# ---- 10 ---------------------------------------------------------------------------------------------------
def check_no_trade_days(df: pd.DataFrame) -> dict:
    prev = df.groupby("ticker")["close"].shift(1)
    z = df["volume"] == 0
    moved = z & prev.notna() & (df["close"] != prev)
    return {
        "change_negative_rows": int((df["change"] < 0).sum()), "change_positive_rows": int((df["change"] > 0).sum()),
        "sign": "음수 부호가 붙은 정수 문자열 (예: '-400')",
        "zero_volume_rows": int(z.sum()),
        "zero_volume_open_high_low_zero": int((z & (df["open"] == 0) & (df["high"] == 0) & (df["low"] == 0)).sum()),
        "zero_volume_close_zero": int((z & (df["close"] == 0)).sum()),
        "zero_volume_close_eq_prev": int((z & (df["close"] == prev)).sum()),
        "zero_volume_close_moved": int(moved.sum()),
        "zero_volume_close_moved_common": int((moved & df["common"]).sum()),
        "zero_volume_close_moved_by_year": df.loc[moved, "date"].dt.year.value_counts().sort_index().to_dict(),
        "base_eq_prev_close_on_moved": int((moved & (df["base"] == prev)).sum()),
        "close_zero_any": int((df["close"] == 0).sum()),
        "verdict": "무거래일: 시가·고가·저가 0, 종가는 직전 종가 그대로. 일부(주로 2010~2014 비유동 우선주)는 "
                   "체결 없이 종가가 바뀐다(기세로 추정). 기준가 = 직전 종가는 그날도 성립",
    }


# ---- 1 ----------------------------------------------------------------------------------------------------
def check_delisting(df: pd.DataFrame, hold: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    dates = pd.DatetimeIndex(sorted(df["date"].unique()))
    pos = pd.Series(np.arange(len(dates)), index=dates)
    last_day = dates[-1]
    g = df.groupby("ticker").agg(first=("date", "min"), last=("date", "max"), n=("date", "size"),
                                 name=("name", "last"), market=("market", "last"), common=("common", "last"))
    g["missing_days"] = (g["last"].map(pos) - g["first"].map(pos) + 1) - g["n"]
    dl = g[g["last"] < last_day].copy()
    k = df.groupby("ticker").cumcount(ascending=False)          # 0 = 마지막 날
    r = df["close"] / df["base"] - 1
    tail = pd.DataFrame({"ticker": df["ticker"], "k": k, "absr": r.abs(), "vol": df["volume"]})
    tail = tail[tail["ticker"].isin(dl.index[dl["common"]]) & (tail["k"] < 10)]

    def zero_run(v):
        v = v.to_numpy()[::-1]
        return int(np.argmax(v != 0)) if (v != 0).any() else len(v)
    runs = df[df["ticker"].isin(dl.index[dl["common"]])].groupby("ticker")["volume"].apply(zero_run)
    dl["zero_volume_tail"] = runs
    h = hold.merge(dl[["last"]], left_on="ticker", right_index=True)
    in_u = h[(h["last"] >= h["eff"]) & (h["last"] < h["end"])]
    dlc = dl[dl["common"]]
    summary = {
        "last_trading_day": str(last_day.date()),
        "tickers_with_missing_days_inside_span": int((g["missing_days"] > 0).sum()),
        "delisted_all": len(dl), "delisted_common": len(dlc),
        "delisted_common_by_year_market": {f"{y}": d for y, d in
                                           dlc.groupby([dlc["last"].dt.year, "market"]).size().unstack(fill_value=0)
                                           .to_dict(orient="index").items()},
        "delisted_in_main_universe": len(in_u),
        "delisted_in_main_universe_by_year": in_u["last"].dt.year.value_counts().sort_index().to_dict(),
        "tail_median_abs_return_by_k": tail.groupby("k")["absr"].median().round(4).to_dict(),
        "tail_share_abs_return_gt_30pct_by_k": tail.groupby("k")["absr"].apply(lambda s: (s > 0.3).mean())
        .round(3).to_dict(),
        "delisted_common_ending_in_halt": int((runs > 0).sum()),
        "verdict": "거래정지 기간에도 응답에서 빠지지 않는다(거래량 0) → 빠진 날 0, gap 후보 없음. "
                   "마지막 7거래일이 정리매매(첫날 |수익률| 중앙값 큼), 그 전은 거래정지인 패턴",
    }
    return summary, dl.reset_index()


# ---- 2 ----------------------------------------------------------------------------------------------------
def resume_windows(df: pd.DataFrame) -> pd.DataFrame:
    """재개일(resume)마다 직전 마지막 거래일부터 재개일까지 묶어 수정 비율을 비교한다.

    adj  = 구간 안 기준가 비율(prev_close / base)의 곱. 기본 공식이 '수정'으로 처리한 몫
    k_sh = 재개일 주식수 / 정지 직전 거래일 주식수. 분할·병합·감자만 있었다면 adj ≈ k_sh
    dev  = log(adj / k_sh). > 0 이면 기준가가 주식수로 수정한 직전 종가보다 낮게 잡혔다
           (기본 공식이 그 하락을 수익률로 세지 않는다)
    """
    d = df.reset_index(drop=True)
    traded_idx = pd.Series(np.where(d["volume"] > 0, d.index, np.nan))
    d["last_traded"] = traded_idx.groupby(d["ticker"]).shift(1)
    d["last_traded"] = d["last_traded"].groupby(d["ticker"]).ffill()
    logr = np.log(d["ratio"].astype(float))
    d["cum_logr"] = logr.groupby(d["ticker"]).cumsum()
    ev = d[d["resume"] & d["last_traded"].notna()].copy()
    j = ev["last_traded"].astype(int).to_numpy()
    ev["start_date"] = d.loc[j, "date"].to_numpy()
    ev["halt_days"] = ev.index.to_numpy() - j - 1
    ev["P0"] = d.loc[j, "close"].to_numpy().astype(float)
    ev["sh0"] = d.loc[j, "shares"].to_numpy().astype(float)
    ev["par0"] = d.loc[j, "parval"].to_numpy().astype(float)
    ev["adj"] = np.exp(ev["cum_logr"].to_numpy() - d.loc[j, "cum_logr"].to_numpy())
    ev["k_sh"] = ev["shares"].astype(float) / ev["sh0"]
    ev["dev"] = np.log(ev["adj"] / ev["k_sh"])
    sh_same = (ev["k_sh"] - 1).abs() < 1e-9
    ev["cls"] = np.select(
        [sh_same & ((ev["adj"] - 1).abs() < 1e-9), ~sh_same & (ev["dev"].abs() < TOL), sh_same],
        ["no_adjustment", "shares_explain", "reset_shares_same"], "shares_dont_explain")
    return ev[["date", "ticker", "name", "market", "common", "start_date", "halt_days", "P0", "base", "close",
               "sh0", "shares", "par0", "parval", "adj", "k_sh", "dev", "cls"]].reset_index(drop=True)


def check_resume(df: pd.DataFrame, hold: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    ev = resume_windows(df)
    ev = ev[ev["common"]]
    h = hold.merge(ev[["ticker", "date"]], on="ticker")
    in_u = h[(h["date"] >= h["eff"]) & (h["date"] < h["end"])][["ticker", "date"]].drop_duplicates()
    ev = ev.merge(in_u.assign(in_main=True), on=["ticker", "date"], how="left").fillna({"in_main": False})
    need = ev["cls"].isin(["reset_shares_same", "shares_dont_explain"])
    split = df[(df["ratio"] != 1) & ~df["resume"] & ~df["halt_adjust"] & ~df["listing"] & df["common"]]
    summary = {
        "events_ratio_ne_1_common": int(((df["ratio"] != 1) & ~df["listing"] & df["common"]).sum()),
        "events_while_trading_common": len(split),
        "halt_adjust_days_common": int((df["halt_adjust"] & df["common"]).sum()),
        "resume_events_common": len(ev),
        "resume_cls": ev["cls"].value_counts().to_dict(),
        "resume_cls_in_main_universe": ev.loc[ev["in_main"], "cls"].value_counts().to_dict(),
        "need_k": int(need.sum()), "need_k_in_main_universe": int((need & ev["in_main"]).sum()),
        "need_k_in_main_dev_positive": int((need & ev["in_main"] & (ev["dev"] > 0)).sum()),
        "need_k_abs_dev_quantiles": ev.loc[need, "dev"].abs().quantile([0.5, 0.9]).round(3).to_dict(),
        "verdict": "거래 중 분할·병합·권리락은 기준가가 바로 반영해 기본 공식이 맞다. 거래정지 뒤 재개일 중 "
                   "주식수 변화로 설명되지 않는 사건(감자+출자전환, 인적분할, 장기 정지 뒤 기준가 재평가)은 "
                   "k를 권리일정·OpenDART에서 구해야 한다",
    }
    return summary, ev


# ---- 3 ----------------------------------------------------------------------------------------------------
def check_universe(stats: pd.DataFrame, members: pd.DataFrame, df: pd.DataFrame, hold: pd.DataFrame) -> dict:
    pp = df[["date", "ticker", "volume"]].merge(hold, on="ticker")
    pp = pp[(pp["date"] >= pp["eff"]) & (pp["date"] < pp["end"])]
    z = pp.groupby(["eff", "ticker"])["volume"].apply(lambda v: (v == 0).mean())
    sens_tickers = set(members.loc[members["universe"] == "sens", "ticker"])
    return {
        "rebalances": len(stats),
        "main_filled_every_time": bool((stats["main"] == uni.MAIN_SIZE).all()),
        "sens_filled_every_time": bool((stats["sens"] == uni.SENS_SIZE).all()),
        "remain_after_filters_min": int(stats["remain"].min()),
        "per_rebalance": stats.assign(eff=stats["eff"].dt.strftime("%Y-%m-%d"), base=stats["base"].dt.strftime("%Y-%m-%d"))
        [["eff", "base", "eligible", "spac_out", "val20_missing", "liq_out", "halt_out", "admin_out", "remain",
          "main", "main_kosdaq", "sens"]].to_dict(orient="records"),
        "main_kosdaq_share_range": [round(float((stats["main_kosdaq"] / stats["main"]).min()), 3),
                                    round(float((stats["main_kosdaq"] / stats["main"]).max()), 3)],
        "member_periods": int(len(z)), "member_periods_any_halt_day": int((z > 0).sum()),
        "member_periods_halt_gt_10pct": int((z > 0.1).sum()), "member_periods_halt_gt_50pct": int((z > 0.5).sum()),
        "member_day_halt_share": round(float((pp["volume"] == 0).mean()), 5),
        "sens_unique_tickers": len(sens_tickers),
    }


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    return o


def run(write: bool = True) -> dict:
    data = load()
    df = data["df"]
    members, stats = uni.build(write=write)
    hold = uni.holdings(members, df["date"].max())
    report = {"item4_first_day": check_first_day(),
              "item7_info_snapshot": check_info_snapshot(data["prices"], data["info"]),
              "item8_sect": check_sect(data["prices"]),
              "item9_ticker_format": check_ticker_format(df),
              "item10_no_trade_days": check_no_trade_days(df)}
    report["item1_delisting"], delisted = check_delisting(df, hold)
    report["item2_resume"], resume = check_resume(df, hold)
    report["item3_universe"] = check_universe(stats, members, df, hold)
    if write:
        out = out_dir()
        out.mkdir(parents=True, exist_ok=True)
        (out / "stageA_data.json").write_text(json.dumps(_jsonable(report), ensure_ascii=False, indent=1, default=str),
                                              encoding="utf-8")
        delisted.to_parquet(out / "delisted.parquet", index=False)
        resume.to_parquet(out / "resume_events.parquet", index=False)
    return report


def main() -> None:
    report = run()
    for k, v in report.items():
        print(k, "→", v.get("verdict", "") if isinstance(v, dict) else "")
    print("저장:", out_dir())


if __name__ == "__main__":
    main()
