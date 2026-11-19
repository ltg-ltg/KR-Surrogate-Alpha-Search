"""이미 받아 둔 KRX 원본을 연구 데이터 폴더로 복사한다 (연구 기간 안 파일만).

    python -m krsas.ingest.import_local --src C:\\kr_quant_machine\\data\\raw\\krx --via kr_quant_machine@84c811d

- 서비스 4개(krsas.ingest.krx.SERVICES)의 DATA_START~DATA_END 파일만 복사한다. 2024년 이후 파일은 건너뛴다
- 파일 내용은 바꾸지 않는다 (API 응답 그대로). 수정 시각을 보존하고 그 값을 수집일로 기록한다
- 이미 같은 해시로 들어와 있는 파일은 다시 복사하지 않는다. 해시가 다르면 멈춘다 (raw는 고치지 않는다)
- 휴장일 달력(_calendar.json)도 연구 기간 안 날짜만 옮긴다
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

from ..paths import DATA_END, DATA_START, krx_raw_dir
from . import manifest
from .krx import SERVICES


def import_local(src: Path, via: str) -> dict:
    dst_root = krx_raw_dir()
    known = manifest.load()
    stats = {"copied": 0, "already": 0, "skipped_out_of_window": 0}
    new_entries = []
    for service in SERVICES:
        for f in sorted((src / service).glob("*/*.json.gz")):
            bas_dd = f.name[:8]
            if not DATA_START <= bas_dd <= DATA_END:
                stats["skipped_out_of_window"] += 1
                continue
            dst = dst_root / service / bas_dd[:4] / f.name
            rel = dst.relative_to(dst_root).as_posix()
            if dst.exists():
                if rel in known and known[rel]["sha256"] == manifest.sha256(f):
                    stats["already"] += 1
                    continue
                raise RuntimeError(f"{rel}가 이미 있는데 원본과 다릅니다. raw는 덮어쓰지 않습니다")
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst)
            collected = datetime.fromtimestamp(f.stat().st_mtime)
            new_entries.append(manifest.entry(dst, service, bas_dd, collected_at=collected, via=via))
            stats["copied"] += 1
            if len(new_entries) >= 500:
                manifest.append(new_entries)
                new_entries = []
    manifest.append(new_entries)

    cal_src = src / "_calendar.json"
    if cal_src.exists():
        cal = json.loads(cal_src.read_text(encoding="utf-8"))
        cal = {d: v for d, v in sorted(cal.items()) if DATA_START <= d <= DATA_END}
        (dst_root / "_calendar.json").write_text(json.dumps(cal), encoding="utf-8")
        stats["calendar_days"] = len(cal)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--via", required=True, help="수집 경로 기록 (예: kr_quant_machine@84c811d)")
    args = parser.parse_args()
    print(import_local(args.src, args.via))
    check = manifest.verify()
    print({k: (len(v) if isinstance(v, list) else v) for k, v in check.items()})


if __name__ == "__main__":
    main()
