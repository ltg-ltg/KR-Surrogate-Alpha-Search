"""공공데이터포털 금융위원회_주식권리일정정보 원본 수집 (설계서 5절 보조 공급원, 5.5절 재설정일 판정).

    python -m krsas.ingest.rights                 # 오늘 예산이 남는 만큼
    python -m krsas.ingest.rights --max-days 5

- 엔드포인트: https://apis.data.go.kr/1160100/GetStocRighScheService_V2/getRighExerReasSche_V2
  (상세 페이지 data.go.kr/data/15059609에는 오퍼레이션 이름이 없다. 2026-11-19 시험 호출로 확인)
- 키: .env의 DATA_GO_KR_API (일반 인증키 Decoding). 이용 조건: 공공누리 제2유형 (출처 표시, 상업적 이용 금지),
  원천 한국예탁결제원
- basDt = 기업 행위의 기준일. 한 행위가 rgtExertRcd(권리행사사유)별 여러 행으로 나온다
  (01 기준일, 03 권리락일, 09 상장일 …). stckIssuRcd는 주식발행사유 (101 유상증자, 201 액면분할, 202 액면병합 …)
  예: 삼성전자 액면분할 basDt 20180502, 09 상장일 20180504, stckParPrc 100 (새 액면가)
- 배정 비율 필드는 없다. 분할·병합 비율은 액면가 변화로, 증자·감자 비율은 상장주식수 변화나 OpenDART로 구한다
- 종목코드가 없다. crno(법인등록번호)와 회사명으로 매핑한다 (OpenDART 기업개황 jurir_no)
- 대상: DATA_START(20100104) ~ DATA_END의 모든 달력일 (basDt는 휴일일 수도 있다). 2024년 이후 basDt는 받지 않는다
- 저장: {data_root}/raw/rights/{YYYY}/{YYYYMMDD}.json.gz (그날 모든 페이지의 item을 합친 list, 응답 그대로)
  행이 0개인 날도 빈 list로 저장해 다시 부르지 않는다. 원본 목록은 raw/rights/_manifest.jsonl
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from ..paths import DATA_END, DATA_START, ROOT, check_date, data_root
from .krx import BudgetExceeded, CallBudget
from .manifest import sha256

log = logging.getLogger("rights")
ENDPOINT = "https://apis.data.go.kr/1160100/GetStocRighScheService_V2/getRighExerReasSche_V2"
PAGE_SIZE = 1000
# 공공데이터포털 게이트웨이 오류 중 다시 부르면 되는 것: 01 어플리케이션 에러, 04 HTTP 에러, 05 서비스 연결 실패
# (2026-11-19: 20101231 p16에서 HTTP 200 + 04를 받았다). 22는 일일 한도 초과라 예산 소진으로 본다
TRANSIENT_CODES = {"01", "04", "05"}
# 2026-11-19: 20101231 p2가 resultCode 00인데 totalCount 0·빈 item으로 왔다 (다시 부르면 35,134건 정상).
# 페이지마다 totalCount와 행 수가 첫 페이지 기준과 맞는지 보고, 어긋나면 다시 부른다
PAGE_RETRIES = 5
DAILY_LIMIT = 9500          # 개발계정 일 10,000회 가정에서 여유를 둔다 `[확인: 마이페이지 트래픽]`


def rights_dir() -> Path:
    return data_root() / "raw" / "rights"


def raw_path(bas_dd: str) -> Path:
    check_date(bas_dd)
    return rights_dir() / bas_dd[:4] / f"{bas_dd}.json.gz"


def read_raw(bas_dd: str) -> list[dict]:
    with gzip.open(raw_path(bas_dd), "rt", encoding="utf-8") as fh:
        return json.load(fh)


def target_days() -> list[str]:
    return [d.strftime("%Y%m%d") for d in pd.date_range(DATA_START, DATA_END)]


class RightsClient:
    def __init__(self, api_key: str | None = None, pause: float = 0.1, budget: CallBudget | None = None):
        load_dotenv(ROOT / ".env")
        self.api_key = (api_key or os.environ.get("DATA_GO_KR_API") or "").strip()
        if not self.api_key:
            raise RuntimeError(".env에 DATA_GO_KR_API가 없습니다")
        self.pause = pause
        self.budget = budget
        self.session = requests.Session()

    def _page(self, bas_dd: str, page: int) -> dict:
        last_exc: Exception | None = None
        for attempt in range(4):
            try:
                if self.budget is not None:
                    self.budget.spend()
                resp = self.session.get(ENDPOINT, params={
                    "serviceKey": self.api_key, "resultType": "json", "numOfRows": PAGE_SIZE,
                    "pageNo": page, "basDt": bas_dd}, timeout=60)
                time.sleep(self.pause)
                resp.encoding = "utf-8"
                if resp.status_code >= 500 and attempt < 3:
                    time.sleep(2 ** attempt)
                    continue
                data = resp.json()
                if "response" not in data:          # 게이트웨이 오류는 OpenAPI_ServiceResponse로 온다 (HTTP 200일 때도 있다)
                    code = data.get("OpenAPI_ServiceResponse", {}).get("cmmMsgHeader", {}).get("returnReasonCode")
                    if code == "22":
                        raise BudgetExceeded(f"권리일정 일일 호출 한도 초과 ({bas_dd} p{page})")
                    if code in TRANSIENT_CODES and attempt < 3:
                        last_exc = RuntimeError(f"게이트웨이 오류 {code}")
                        time.sleep(2 ** (attempt + 1))
                        continue
                    raise RuntimeError(f"권리일정 {bas_dd} p{page}: {json.dumps(data, ensure_ascii=False)[:300]}")
                header = data["response"]["header"]
                if header.get("resultCode") != "00":
                    raise RuntimeError(f"권리일정 {bas_dd} p{page}: {header}")
                return data["response"]["body"]
            except (requests.RequestException, ValueError) as exc:
                last_exc = exc
                time.sleep(2 ** attempt)
        raise RuntimeError(f"권리일정 요청 실패: {bas_dd} p{page}") from last_exc

    def fetch(self, bas_dd: str) -> list[dict]:
        """기준일의 모든 행. 페이지를 끝까지 넘긴다."""
        check_date(bas_dd)

        def parse(body):
            items = (body.get("items") or {}).get("item") or []
            return int(body.get("totalCount") or 0), [items] if isinstance(items, dict) else items

        total, rows = parse(self._page(bas_dd, 1))
        if total == 0:                      # 서버가 정상 코드로 빈 응답을 줄 때가 있다. 0건은 한 번 더 확인한다
            total, rows = parse(self._page(bas_dd, 1))
        page = 1
        while len(rows) < total:
            page += 1
            expected = min(PAGE_SIZE, total - len(rows))
            for attempt in range(PAGE_RETRIES):
                t, items = parse(self._page(bas_dd, page))
                if t == total and len(items) == expected:
                    break
                log.warning("권리일정 %s p%d: totalCount %d·%d행 (기대 %d·%d). 다시 부른다",
                            bas_dd, page, t, len(items), total, expected)
                time.sleep(2 ** attempt)
            else:
                raise RuntimeError(f"권리일정 {bas_dd} p{page}: 응답이 {PAGE_RETRIES}번 모두 어긋났습니다")
            rows.extend(items)
        if len(rows) != total:
            raise RuntimeError(f"권리일정 {bas_dd}: {len(rows)}행 != totalCount {total}")
        return rows


def _write(bas_dd: str, rows: list[dict]) -> Path:
    path = raw_path(bas_dd)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False)
    tmp.replace(path)
    return path


def collect(client: RightsClient, *, max_days: int | None = None) -> dict:
    stats = {"days_done": 0, "rows": 0, "stopped": None}
    pending: list[dict] = []
    manifest = rights_dir() / "_manifest.jsonl"
    try:
        for bas_dd in target_days():
            if raw_path(bas_dd).exists():
                continue
            if max_days is not None and stats["days_done"] >= max_days:
                stats["stopped"] = "max_days"
                break
            rows = client.fetch(bas_dd)
            path = _write(bas_dd, rows)
            pending.append({"bas_dd": bas_dd, "path": path.relative_to(rights_dir()).as_posix(),
                            "rows": len(rows), "sha256": sha256(path), "bytes": path.stat().st_size,
                            "endpoint": ENDPOINT, "collected_at": datetime.now().isoformat(timespec="seconds")})
            stats["days_done"] += 1
            stats["rows"] += len(rows)
            if stats["days_done"] % 100 == 0:
                log.info("  %d일 완료 (마지막 %s), 남은 예산 %d", stats["days_done"], bas_dd,
                         client.budget.remaining if client.budget else -1)
    except BudgetExceeded:
        stats["stopped"] = "budget"
    finally:
        if pending:
            manifest.parent.mkdir(parents=True, exist_ok=True)
            with open(manifest, "a", encoding="utf-8") as fh:
                for e in pending:
                    fh.write(json.dumps(e, ensure_ascii=False) + "\n")
        if client.budget is not None:
            client.budget.flush()
    stats["remaining_days"] = sum(1 for d in target_days() if not raw_path(d).exists())
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-days", type=int, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    budget = CallBudget(DAILY_LIMIT, rights_dir() / "_call_budget.json")
    client = RightsClient(budget=budget)
    log.info("시작. 남은 예산 %d", budget.remaining)
    log.info("끝: %s", collect(client, max_days=args.max_days))


if __name__ == "__main__":
    main()
