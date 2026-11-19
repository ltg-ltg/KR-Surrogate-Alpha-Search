"""원본 목록: 파일마다 해시, 크기, 수집일, 엔드포인트, 수집 경로를 적는다 (설계서 5절 "기록").

{data_root}/raw/krx/_manifest.jsonl에 한 줄씩 쌓는다. 같은 파일이 다시 기록되면 마지막 줄이 유효하다.
해시 목록은 KRX 데이터가 아니므로 공개해도 된다 (재현자가 같은 원본을 받았는지 대조하는 용도).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

from ..paths import krx_raw_dir
from .krx import BASE_URL, SERVICES

SPEC_VERSION = None   # KRX Open API 개발 명세서 버전. 확인 후 채운다 (체크리스트 2절)


def manifest_path() -> Path:
    return krx_raw_dir() / "_manifest.jsonl"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def entry(path: Path, service: str, bas_dd: str, *, collected_at: datetime, via: str) -> dict:
    return {
        "service": service, "bas_dd": bas_dd,
        "path": path.relative_to(krx_raw_dir()).as_posix(),
        "sha256": sha256(path), "bytes": path.stat().st_size,
        "endpoint": f"{BASE_URL}/{SERVICES[service]}", "spec_version": SPEC_VERSION,
        "collected_at": collected_at.isoformat(timespec="seconds"), "via": via,
    }


def append(entries: list[dict]) -> None:
    if not entries:
        return
    path = manifest_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        for e in entries:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")


def load() -> dict[str, dict]:
    """{상대 경로: 마지막 기록}"""
    path = manifest_path()
    out: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                e = json.loads(line)
                out[e["path"]] = e
    return out


def verify() -> dict:
    """데이터 폴더의 원본을 목록과 대조한다. 연구 기간 밖 파일도 찾는다."""
    from ..paths import DATA_END, DATA_START
    known = load()
    stats = {"files": 0, "ok": 0, "hash_mismatch": [], "unlisted": [], "missing": [], "out_of_window": []}
    seen = set()
    for service in SERVICES:
        for f in sorted((krx_raw_dir() / service).glob("*/*.json.gz")):
            rel = f.relative_to(krx_raw_dir()).as_posix()
            seen.add(rel)
            stats["files"] += 1
            if not DATA_START <= f.name[:8] <= DATA_END:
                stats["out_of_window"].append(rel)
            if rel not in known:
                stats["unlisted"].append(rel)
            elif known[rel]["sha256"] != sha256(f):
                stats["hash_mismatch"].append(rel)
            else:
                stats["ok"] += 1
    stats["missing"] = sorted(set(known) - seen)
    return stats
