"""KRX Open API 클라이언트와 원본 저장 규칙.

- 키: 레포 루트 .env의 KRX_OPEN_API (요청 헤더 AUTH_KEY)
- 서비스: 설계서 5절의 4개만. 날짜(basDd)별 한 번 호출로 그날 해당 시장 전 종목 단면을 받는다
- 저장: {data_root}/raw/krx/{service}/{YYYY}/{YYYYMMDD}.json.gz (응답의 OutBlock_1 그대로). raw는 고치지 않는다
- 이용 조건 (설계서 5.0절): 키당 하루 10,000회, 제3자 제공 금지, 결과물에 "KRX 통계정보" 표시, 비상업적 이용

수집 코드는 kr_quant_machine `src/ingest/krx.py`(커밋 84c811d)에서 가져와 연구 기간과 서비스를 좁혔다.
"""
from __future__ import annotations

import gzip
import json
import os
import time
from datetime import date
from pathlib import Path

import requests
from dotenv import load_dotenv

from ..paths import ROOT, check_date, krx_raw_dir

BASE_URL = "https://data-dbg.krx.co.kr/svc/apis"
DAILY_LIMIT = 9500          # 한도 10,000회에서 여유를 둔다. 같은 키를 다른 프로젝트와 함께 쓰면 그 사용량도 한도에 들어간다

SERVICES = {
    "stk_trd": "sto/stk_bydd_trd",            # 유가증권 일별매매정보
    "ksq_trd": "sto/ksq_bydd_trd",            # 코스닥 일별매매정보
    "stk_info": "sto/stk_isu_base_info",      # 유가증권 종목기본정보
    "ksq_info": "sto/ksq_isu_base_info",      # 코스닥 종목기본정보
}
CALENDAR_SERVICE = "stk_trd"                  # 평일마다 먼저 불러 휴장일을 판정한다


def raw_path(service: str, bas_dd: str) -> Path:
    check_date(bas_dd)
    return krx_raw_dir() / service / bas_dd[:4] / f"{bas_dd}.json.gz"


def read_raw(service: str, bas_dd: str) -> list[dict]:
    with gzip.open(raw_path(service, bas_dd), "rt", encoding="utf-8") as fh:
        return json.load(fh)


def write_raw(service: str, bas_dd: str, rows: list[dict]) -> Path:
    path = raw_path(service, bas_dd)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False)
    tmp.replace(path)
    return path


class BudgetExceeded(RuntimeError):
    pass


class CallBudget:
    """하루 호출 수를 파일에 누적해 여러 실행에 걸쳐 한도를 지킨다."""

    def __init__(self, daily_limit: int = DAILY_LIMIT, path: Path | None = None):
        self.daily_limit = daily_limit
        self.path = path or krx_raw_dir() / "_call_budget.json"
        self.day = date.today().isoformat()
        self.used = 0
        if self.path.exists():
            saved = json.loads(self.path.read_text(encoding="utf-8"))
            if saved.get("day") == self.day:
                self.used = int(saved.get("used", 0))

    @property
    def remaining(self) -> int:
        return self.daily_limit - self.used

    def spend(self) -> None:
        if self.remaining <= 0:
            self.flush()
            raise BudgetExceeded(f"오늘({self.day}) 호출 예산 {self.daily_limit}회를 다 썼습니다")
        self.used += 1
        if self.used % 25 == 0:
            self.flush()

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"day": self.day, "used": self.used}), encoding="utf-8")


class KrxError(RuntimeError):
    def __init__(self, service: str, http_status: int, body: str):
        super().__init__(f"KRX {service} HTTP {http_status}: {body[:200]}")
        self.service = service
        self.http_status = http_status


class KrxClient:
    def __init__(self, api_key: str | None = None, pause: float = 0.2,
                 budget: CallBudget | None = None):
        load_dotenv(ROOT / ".env")
        self.api_key = api_key or os.environ.get("KRX_OPEN_API")
        if not self.api_key:
            raise RuntimeError(".env에 KRX_OPEN_API가 없습니다")
        self.pause = pause
        self.budget = budget
        self.session = requests.Session()
        self.session.headers["AUTH_KEY"] = self.api_key.strip()

    def fetch(self, service: str, bas_dd: str) -> list[dict]:
        """기준일(YYYYMMDD)의 전 종목 단면. 휴장일이면 빈 list. 연구 기간 밖 날짜는 부르지 않는다."""
        check_date(bas_dd)
        path = SERVICES[service]
        last_exc: Exception | None = None
        for attempt in range(4):
            try:
                if self.budget is not None:
                    self.budget.spend()
                resp = self.session.get(f"{BASE_URL}/{path}", params={"basDd": bas_dd}, timeout=60)
                time.sleep(self.pause)
                if resp.status_code != 200:
                    if resp.status_code >= 500 and attempt < 3:
                        time.sleep(2 ** attempt)
                        continue
                    raise KrxError(service, resp.status_code, resp.text)
                return resp.json().get("OutBlock_1", [])
            except requests.RequestException as exc:
                last_exc = exc
                time.sleep(2 ** attempt)
        raise RuntimeError(f"KRX 요청 실패: {service} {bas_dd}") from last_exc
