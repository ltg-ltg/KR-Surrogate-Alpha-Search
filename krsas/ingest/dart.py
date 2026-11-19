"""OpenDART 고유번호와 기업개황 수집 → 종목코드·법인등록번호(crno) 대응표 (설계서 5절 보조 공급원, 5.5절).

    python -m krsas.ingest.dart              # corpCode + 보통주 종목의 기업개황 (남은 것만)
    python -m krsas.ingest.dart --refresh-corpcode

권리일정정보(krsas.ingest.rights)에는 종목코드가 없고 crno(법인등록번호)만 있다. 이 모듈이 만든 대응표로
권리일정 행을 종목에 붙인다.

- corpCode.xml: 전체 고유번호(corp_code)와 종목코드. 상장폐지 회사도 마지막 종목코드를 갖고 있다
  (2026-11-19: 연구 기간 보통주 3,051종목이 모두 stock_code로 대응됨). 우선주 코드는 없다
- company.json: corp_code별 기업개황. jurir_no가 법인등록번호(= 권리일정 crno)
- 키: .env의 OPEN_DART_API. 개인 키 하루 20,000건 한도 (DAILY_LIMIT은 여유를 둔 값)
- 저장: {data_root}/raw/dart/corpCode.zip, raw/dart/company/{corp_code}.json (응답 그대로),
  raw/dart/_manifest.jsonl. 대응표는 {data_root}/clean/dart_corp.parquet
- 종목코드 재사용: corpCode의 stock_code는 회사의 마지막 코드라, 같은 코드를 예전에 다른 회사가 썼으면 틀린다.
  대응표에 KRX 종목명과 DART 종목명(stock_name)의 일치 여부를 남겨 확인한다
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import os
import time
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from ..paths import ROOT, data_root
from .clean import clean_dir
from .krx import BudgetExceeded, CallBudget
from .manifest import sha256

log = logging.getLogger("dart")
BASE = "https://opendart.fss.or.kr/api"
DAILY_LIMIT = 19_000
NO_DATA = "013"
LIMIT_CODE = "020"          # 요청 제한 초과


def dart_dir() -> Path:
    return data_root() / "raw" / "dart"


def company_path(corp_code: str) -> Path:
    return dart_dir() / "company" / f"{corp_code}.json"


class DartClient:
    def __init__(self, api_key: str | None = None, pause: float = 0.1, budget: CallBudget | None = None):
        load_dotenv(ROOT / ".env")
        self.api_key = (api_key or os.environ.get("OPEN_DART_API") or "").strip()
        if not self.api_key:
            raise RuntimeError(".env에 OPEN_DART_API가 없습니다")
        self.pause = pause
        self.budget = budget
        self.session = requests.Session()

    def get(self, endpoint: str, **params) -> requests.Response:
        last_exc: Exception | None = None
        for attempt in range(4):
            try:
                if self.budget is not None:
                    self.budget.spend()
                resp = self.session.get(f"{BASE}/{endpoint}", params={"crtfc_key": self.api_key, **params},
                                        timeout=120)
                time.sleep(self.pause)
                if resp.status_code >= 500 and attempt < 3:
                    time.sleep(2 ** attempt)
                    continue
                resp.raise_for_status()
                return resp
            except requests.RequestException as exc:
                last_exc = exc
                time.sleep(2 ** attempt)
        raise RuntimeError(f"OpenDART 요청 실패: {endpoint} {params}") from last_exc

    def json(self, endpoint: str, **params) -> dict:
        data = self.get(endpoint, **params).json()
        status = data.get("status")
        if status == LIMIT_CODE:
            raise BudgetExceeded(f"OpenDART 요청 제한 초과: {data.get('message')}")
        if status not in ("000", NO_DATA):
            raise RuntimeError(f"OpenDART {endpoint} {params}: {data}")
        return data


def _manifest(entries: list[dict]) -> None:
    if not entries:
        return
    path = dart_dir() / "_manifest.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        for e in entries:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")


def _entry(path: Path, endpoint: str, key: str) -> dict:
    return {"path": path.relative_to(dart_dir()).as_posix(), "endpoint": endpoint, "key": key,
            "sha256": sha256(path), "bytes": path.stat().st_size,
            "collected_at": datetime.now().isoformat(timespec="seconds")}


def fetch_corpcode(client: DartClient, refresh: bool = False) -> Path:
    path = dart_dir() / "corpCode.zip"
    if path.exists() and not refresh:
        return path
    resp = client.get("corpCode.xml")
    if not resp.content.startswith(b"PK"):
        raise RuntimeError(f"corpCode.xml 응답이 zip이 아닙니다: {resp.content[:200]!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(resp.content)
    _manifest([_entry(path, "corpCode.xml", "all")])
    return path


def read_corpcode(path: Path | None = None) -> pd.DataFrame:
    path = path or dart_dir() / "corpCode.zip"
    with zipfile.ZipFile(path) as z:
        root = ET.fromstring(z.read(z.namelist()[0]))
    rows = [{c.tag: (c.text or "").strip() for c in e} for e in root]
    return pd.DataFrame(rows)


def common_tickers() -> pd.DataFrame:
    """연구 기간 보통주(주권) 종목과 마지막 KRX 종목명."""
    info = pd.read_parquet(clean_dir() / "share_info", columns=["date", "ticker", "name", "stock_type", "secugrp"])
    com = info[(info["stock_type"] == "보통주") & (info["secugrp"] == "주권")]
    return com.sort_values("date").groupby("ticker").agg(krx_name=("name", "last"), last_date=("date", "max"),
                                                        names=("name", lambda s: "|".join(s.unique())))


def collect_company(client: DartClient, corp_codes: list[str]) -> dict:
    stats = {"done": 0, "skipped": 0, "stopped": None}
    pending = []
    try:
        for cc in corp_codes:
            path = company_path(cc)
            if path.exists():
                stats["skipped"] += 1
                continue
            data = client.json("company.json", corp_code=cc)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            pending.append(_entry(path, "company.json", cc))
            stats["done"] += 1
            if stats["done"] % 200 == 0:
                log.info("  기업개황 %d건", stats["done"])
    except BudgetExceeded:
        stats["stopped"] = "budget"
    finally:
        _manifest(pending)
        if client.budget is not None:
            client.budget.flush()
    return stats


def build_map(write: bool = True) -> pd.DataFrame:
    """종목코드 → corp_code, jurir_no(crno). 기업개황이 없는 종목은 crno가 비어 있다."""
    cc = read_corpcode()
    cc = cc[cc["stock_code"] != ""][["stock_code", "corp_code", "corp_name"]]
    t = common_tickers().reset_index().merge(cc, left_on="ticker", right_on="stock_code", how="left")
    rows = []
    for r in t.itertuples():
        info = {}
        if isinstance(r.corp_code, str) and company_path(r.corp_code).exists():
            info = json.loads(company_path(r.corp_code).read_text(encoding="utf-8"))
        rows.append({"ticker": r.ticker, "krx_name": r.krx_name, "krx_names": r.names, "last_date": r.last_date,
                     "corp_code": r.corp_code, "dart_corp_name": info.get("corp_name"),
                     "dart_stock_name": info.get("stock_name"), "crno": (info.get("jurir_no") or "").strip() or None})
    out = pd.DataFrame(rows)
    out["name_match"] = [isinstance(s, str) and s in n.split("|") for s, n in zip(out["dart_stock_name"],
                                                                                    out["krx_names"])]
    if write:
        out.to_parquet(clean_dir() / "dart_corp.parquet", index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-corpcode", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    budget = CallBudget(DAILY_LIMIT, dart_dir() / "_call_budget.json")
    client = DartClient(budget=budget)
    fetch_corpcode(client, refresh=args.refresh_corpcode)
    cc = read_corpcode()
    codes = cc[cc["stock_code"].isin(common_tickers().index)]["corp_code"].unique().tolist()
    log.info("기업개황 대상 %d개, 남은 예산 %d", len(codes), budget.remaining)
    log.info("끝: %s", collect_company(client, codes))
    m = build_map()
    log.info("대응표 %d종목, crno 있음 %d, 이름 일치 %d", len(m), m["crno"].notna().sum(), m["name_match"].sum())


if __name__ == "__main__":
    main()
