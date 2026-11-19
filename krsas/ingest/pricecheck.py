"""단계 A 검증 5: 공공데이터포털 금융위원회_주식시세정보와 KRX 종가·거래량 대조 (설계서 5.3절).

    python -m krsas.ingest.pricecheck          # 표본 날짜 원본 수집 + 대조 → {data_root}/validation/pricecheck.json

- 엔드포인트: GetStockSecuritiesInfoService/getStockPriceInfo (2020-01-02 이후만 제공)
- 표본: 2020~2023 분기마다 첫 거래일 16일. 대조 범위가 OOS와 겹치므로 원자료의 종가·거래량만 맞춘다
  (수식·포트폴리오 성과는 계산하지 않는다)
- 키: .env의 DATA_GO_KR_API_2 (주식시세정보용으로 따로 받은 키). 없으면 DATA_GO_KR_API
- 저장: {data_root}/raw/stockprice/{YYYYMMDD}.json.gz (응답 item 그대로), raw/stockprice/_manifest.jsonl
"""
from __future__ import annotations

import gzip
import json
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from ..paths import ROOT, check_date, data_root
from .clean import clean_dir
from .manifest import sha256

ENDPOINT = "https://apis.data.go.kr/1160100/service/GetStockSecuritiesInfoService/getStockPriceInfo"
PAGE_SIZE = 1000


def raw_dir() -> Path:
    return data_root() / "raw" / "stockprice"


def sample_dates() -> list[str]:
    cal = pd.read_parquet(clean_dir() / "prices", columns=["date"])["date"].drop_duplicates().sort_values()
    cal = cal[cal >= "2020-01-01"]
    first = cal.groupby(cal.dt.to_period("Q")).min()
    return [d.strftime("%Y%m%d") for d in first]


def fetch(bas_dd: str, key: str, session: requests.Session) -> list[dict]:
    check_date(bas_dd)
    rows, page, total = [], 0, None
    while total is None or len(rows) < total:
        page += 1
        for attempt in range(4):
            try:
                resp = session.get(ENDPOINT, params={"serviceKey": key, "resultType": "json", "numOfRows": PAGE_SIZE,
                                                     "pageNo": page, "basDt": bas_dd}, timeout=60)
                data = resp.json()
                if "response" not in data:      # 게이트웨이 오류. 30 = 이 서비스에 활용 신청이 안 된 키
                    hdr = data.get("OpenAPI_ServiceResponse", {}).get("cmmMsgHeader", {})
                    if hdr.get("returnReasonCode") == "30":
                        raise RuntimeError("키가 금융위원회_주식시세정보에 등록되지 않았습니다. "
                                           "공공데이터포털에서 활용 신청 후 다시 실행하세요")
                    raise ValueError(f"게이트웨이 오류 {hdr}")
                body = data["response"]["body"]
                break
            except (requests.RequestException, ValueError, KeyError):
                if attempt == 3:
                    raise
                time.sleep(2 ** (attempt + 1))
        total = int(body.get("totalCount") or 0)
        items = (body.get("items") or {}).get("item") or []
        rows.extend([items] if isinstance(items, dict) else items)
        if not items:
            break
        time.sleep(0.1)
    if len(rows) != total:
        raise RuntimeError(f"주식시세 {bas_dd}: {len(rows)}행 != totalCount {total}")
    return rows


def collect() -> list[str]:
    load_dotenv(ROOT / ".env")
    key = (os.environ.get("DATA_GO_KR_API_2") or os.environ["DATA_GO_KR_API"]).strip()
    session = requests.Session()
    pending = []
    for d in sample_dates():
        path = raw_dir() / f"{d}.json.gz"
        if path.exists():
            continue
        rows = fetch(d, key, session)
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            json.dump(rows, fh, ensure_ascii=False)
        pending.append({"bas_dd": d, "path": path.name, "rows": len(rows), "sha256": sha256(path),
                        "endpoint": ENDPOINT, "collected_at": datetime.now().isoformat(timespec="seconds")})
    if pending:
        with open(raw_dir() / "_manifest.jsonl", "a", encoding="utf-8") as fh:
            for e in pending:
                fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    return sample_dates()


def compare(dates: list[str]) -> dict:
    prices = pd.read_parquet(clean_dir() / "prices", columns=["date", "ticker", "market", "close", "volume"])
    out = {"dates": dates, "per_date": []}
    totals = {"krx_rows": 0, "matched": 0, "close_diff": 0, "volume_diff": 0, "krx_only": 0}
    for d in dates:
        with gzip.open(raw_dir() / f"{d}.json.gz", "rt", encoding="utf-8") as fh:
            pub = pd.DataFrame(json.load(fh))
        pub = pub[pub["mrktCtg"].isin(["KOSPI", "KOSDAQ"])]
        pub = pub.assign(ticker=pub["srtnCd"].str.strip().str[-6:],
                         close_pub=pd.to_numeric(pub["clpr"]), volume_pub=pd.to_numeric(pub["trqu"]))
        k = prices[prices["date"] == pd.Timestamp(d)]
        m = k.merge(pub[["ticker", "close_pub", "volume_pub"]], on="ticker", how="left", indicator=True)
        both = m[m["_merge"] == "both"]
        row = {"date": d, "krx_rows": len(k), "public_rows": len(pub), "matched": len(both),
               "krx_only": int((m["_merge"] == "left_only").sum()),
               "close_diff": int((both["close"] != both["close_pub"]).sum()),
               "volume_diff": int((both["volume"] != both["volume_pub"]).sum())}
        out["per_date"].append(row)
        for key in totals:
            totals[key] += row[key]
    out["totals"] = totals
    return out


def main() -> None:
    dates = collect()
    result = compare(dates)
    path = data_root() / "validation" / "pricecheck.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(result["totals"])
    for r in result["per_date"]:
        print(r)


if __name__ == "__main__":
    main()
