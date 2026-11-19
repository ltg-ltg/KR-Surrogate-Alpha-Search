"""KRX Open API 원본 수집 (재현자용): 연구 기간의 거래일마다 서비스 4개의 전 종목 단면을 받는다.

    python -m src.ingest.collect                # 오늘 예산이 남는 만큼
    python -m src.ingest.collect --max-days 5

- 대상: DATA_START ~ DATA_END 평일. 2024년 이후는 받지 않는다
- 휴장일: 평일마다 CALENDAR_SERVICE를 먼저 부르고, 비어 있으면 _calendar.json에 휴장일로 적은 뒤 나머지를 건너뛴다
- 이미 받은 파일은 다시 받지 않는다. 예산이 떨어지면 멈추고 다음 실행에서 이어받는다
- 받은 파일은 원본 목록(_manifest.jsonl)에 해시와 함께 적는다
- 공개된 원본 해시 목록과의 대조 도구는 해시 공개 시점에 추가한다 (체크리스트 8절)
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime

import pandas as pd

from ..paths import DATA_END, DATA_START, krx_raw_dir
from . import manifest
from .krx import CALENDAR_SERVICE, SERVICES, BudgetExceeded, CallBudget, KrxClient, raw_path, write_raw

log = logging.getLogger("collect")


def calendar_path():
    return krx_raw_dir() / "_calendar.json"


def load_calendar() -> dict[str, bool]:
    p = calendar_path()
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def save_calendar(cal: dict[str, bool]) -> None:
    p = calendar_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(dict(sorted(cal.items()))), encoding="utf-8")
    tmp.replace(p)


def target_days() -> list[str]:
    return [d.strftime("%Y%m%d") for d in pd.bdate_range(DATA_START, DATA_END)]


def collect(client: KrxClient, *, max_days: int | None = None, via: str = "src.ingest.collect") -> dict:
    cal = load_calendar()
    stats = {"days_done": 0, "holidays": 0, "files": 0, "stopped": None}
    pending: list[dict] = []
    order = [CALENDAR_SERVICE] + [s for s in SERVICES if s != CALENDAR_SERVICE]
    try:
        for bas_dd in target_days():
            if cal.get(bas_dd) is False:
                continue
            missing = [s for s in order if not raw_path(s, bas_dd).exists()]
            if not missing:
                cal.setdefault(bas_dd, True)
                continue
            if max_days is not None and stats["days_done"] + stats["holidays"] >= max_days:
                stats["stopped"] = "max_days"
                break
            for service in missing:
                rows = client.fetch(service, bas_dd)
                if service == CALENDAR_SERVICE and not rows:
                    cal[bas_dd] = False
                    stats["holidays"] += 1
                    break
                if not rows:
                    raise RuntimeError(f"거래일인데 {service} {bas_dd} 응답이 비어 있습니다")
                path = write_raw(service, bas_dd, rows)
                pending.append(manifest.entry(path, service, bas_dd, collected_at=datetime.now(), via=via))
                stats["files"] += 1
            else:
                cal[bas_dd] = True
                stats["days_done"] += 1
            if len(pending) >= 200:
                manifest.append(pending)
                pending = []
                save_calendar(cal)
    except BudgetExceeded:
        stats["stopped"] = "budget"
    finally:
        manifest.append(pending)
        save_calendar(cal)
        if client.budget is not None:
            client.budget.flush()
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-days", type=int, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    client = KrxClient(budget=CallBudget())
    log.info("끝: %s", collect(client, max_days=args.max_days))


if __name__ == "__main__":
    main()
