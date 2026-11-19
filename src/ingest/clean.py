"""KRX 원본 → 정제 테이블. 값은 고치지 않고 형만 맞춘다 (원주가 그대로).

    python -m src.ingest.clean [--force]

kr_quant_machine `src/ingest/krx_clean.py`(커밋 351ec2c)에서 가져와 서비스 4개와 연구 기간으로 좁혔다.

연도별 파일로 쓴다 (pd.read_parquet(디렉터리)로 한 번에 읽는다). 원본 파일 수가 바뀐 연도만 다시 만든다.

- {data_root}/clean/prices/{YYYY}.parquet      date, ticker, name, market, sect, open, high, low, close, change, volume, value, mktcap, shares
- {data_root}/clean/share_info/{YYYY}.parquet  date, ticker, isin, name, name_full, list_date, market, secugrp, sect, stock_type, parval, shares

- 종목기본정보 응답에는 기준일이 없어서 파일 이름의 날짜를 붙인다. 무액면 주식의 parval은 NaN이다
- 거래정지일: volume 0, open 0, close는 직전 종가 그대로로 나온다 (2016년 표본 확인)
- 유가증권시장의 sect(SECT_TP_NM)는 빈 문자열이다 (2012·2015·2019·2023 표본). 코스닥만 소속부가 있다
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from ..paths import DATA_END, DATA_START, data_root, krx_raw_dir
from .krx import read_raw

CLEAN_VERSION = 1

TRD_COLS = {"ISU_CD": "ticker", "ISU_NM": "name", "MKT_NM": "market", "SECT_TP_NM": "sect",
            "TDD_OPNPRC": "open", "TDD_HGPRC": "high", "TDD_LWPRC": "low", "TDD_CLSPRC": "close",
            "CMPPREVDD_PRC": "change",          # 기준가 대비. 기준가 = close − change (권리락·분할이 반영된 값)
            "ACC_TRDVOL": "volume", "ACC_TRDVAL": "value", "MKTCAP": "mktcap", "LIST_SHRS": "shares"}
INFO_COLS = {"ISU_SRT_CD": "ticker", "ISU_CD": "isin", "ISU_ABBRV": "name", "ISU_NM": "name_full",
             "LIST_DD": "list_date", "MKT_TP_NM": "market", "SECUGRP_NM": "secugrp", "SECT_TP_NM": "sect",
             "KIND_STKCERT_TP_NM": "stock_type", "PARVAL": "parval", "LIST_SHRS": "shares"}
TEXT = {"ticker", "isin", "name", "name_full", "market", "secugrp", "sect", "stock_type"}

# 테이블 → (서비스들, 열 매핑, 키)
TABLES = {
    "prices": (("stk_trd", "ksq_trd"), TRD_COLS, ["date", "ticker"]),
    "share_info": (("stk_info", "ksq_info"), INFO_COLS, ["date", "ticker"]),
}


def clean_dir():
    return data_root() / "clean"


def to_frame(rows: list[dict], cols: dict[str, str], bas_dd: str) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=list(cols)).rename(columns=cols)
    for col in df.columns:
        if col in TEXT:
            df[col] = df[col].astype("string").str.strip()
        elif col == "list_date":
            df[col] = pd.to_datetime(df[col], format="%Y%m%d", errors="coerce")
        else:
            df[col] = pd.to_numeric(df[col].astype("string").str.replace(",", "", regex=False), errors="coerce")
    df.insert(0, "date", pd.Timestamp(bas_dd))
    return df


def _year_files(service: str, year: str) -> list[str]:
    folder = krx_raw_dir() / service / year
    if not folder.exists():
        return []
    return sorted(d for d in (p.name[:8] for p in folder.glob("*.json.gz")) if DATA_START <= d <= DATA_END)


def build_year(table: str, year: str) -> pd.DataFrame:
    services, cols, key = TABLES[table]
    frames = [to_frame(read_raw(s, d), cols, d) for s in services for d in _year_files(s, year)]
    frames = [f for f in frames if len(f)]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True).sort_values(key, kind="stable").reset_index(drop=True)
    dup = df.duplicated(key).sum()
    if dup:
        raise RuntimeError(f"{table} {year}: 키 중복 {dup}행")
    return df


def build_all(force: bool = False) -> dict:
    out_root = clean_dir()
    ledger_path = out_root / "_clean.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.exists() else {}
    if ledger.get("version") != CLEAN_VERSION:
        ledger = {"version": CLEAN_VERSION, "years": {}}
    stats = {"rebuilt": 0, "rows": {}}
    years = [str(y) for y in range(int(DATA_START[:4]), int(DATA_END[:4]) + 1)]
    for table, (services, _, _) in TABLES.items():
        for year in years:
            n_files = sum(len(_year_files(s, year)) for s in services)
            if not n_files:
                continue
            tag = f"{table}/{year}"
            out = out_root / table / f"{year}.parquet"
            if not force and ledger["years"].get(tag) == n_files and out.exists():
                continue
            df = build_year(table, year)
            if df.empty:
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out, index=False)
            ledger["years"][tag] = n_files
            stats["rebuilt"] += 1
            stats["rows"][tag] = len(df)
    out_root.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    print(build_all(force=parser.parse_args().force))


if __name__ == "__main__":
    main()
