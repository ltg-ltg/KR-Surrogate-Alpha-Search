"""Study 1 수식 풀 `pool_v1` (설계서 7.1절, 단계 B 주 풀).

Study 1 레포는 artifacts/를 git에 넣지 않아 태그만으로는 풀 파일이 따라오지 않는다. 그래서
작성자 환경의 Study 1 산출물을 {data_root}/study1/pool_v1.parquet으로 복사하고, 아래 두 해시로 고정한다.

- FILE_SHA256: 파일 바이트 해시. 복사본이 원본과 같은지 확인한다
- CONTENT_SHA256: (formula_id, formula, hash) 행들의 해시. parquet 메타데이터와 무관하게 내용만 본다.
  Study 1 v1.0-results 코드로 풀을 다시 생성했을 때 같은 풀인지 이 값으로 대조한다
  2026-11-19 확인: Study 1 v1.0-results 코드(로컬 작업 트리, src·config가 태그와 같음)와 Study 1 미국 데이터로
  다시 생성한 풀이 4만 개 모두 같았다 (content_sha256 일치, 해시 차집합 0)

    python -m krsas.pool                 # 복사본 확인
    python -m krsas.pool --compare X     # 다시 생성한 풀 X와 내용 대조

참고: 풀 생성의 시그널 필터는 Study 1 미국 데이터로 계산됐다 (generate.py 문서). valid_ratio·constant_day_ratio
열은 미국 값이고, 한국 시그널 필터(결측·상수 시그널 제외)는 한국 데이터로 다시 계산한다 (설계서 7.1절).
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from .paths import data_root

POOL_SIZE = 40_000
SOURCE = "surrogate-alpha-search artifacts/pool_v1.parquet (v1.0-results 코드로 2026-11-12 생성)"
FILE_SHA256 = "6d9118a0f3802f3358992addc27579cfca44c8b47eecaabc56784401e808a2a8"
CONTENT_SHA256 = "e8d3c00be8f6f318dd6b168dcdd516104c5c5378b37600b78038086c28a2ef6d"


def pool_path() -> Path:
    return data_root() / "study1" / "pool_v1.parquet"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def content_sha256(pool: pd.DataFrame) -> str:
    h = hashlib.sha256()
    for fid, formula, fhash in pool.sort_values("formula_id")[["formula_id", "formula", "hash"]].itertuples(index=False):
        h.update(f"{int(fid)}\t{formula}\t{fhash}\n".encode("utf-8"))
    return h.hexdigest()


def load_pool(verify: bool = True) -> pd.DataFrame:
    path = pool_path()
    if verify and file_sha256(path) != FILE_SHA256:
        raise RuntimeError(f"{path}의 해시가 고정값과 다릅니다")
    pool = pd.read_parquet(path)
    if verify:
        if len(pool) != POOL_SIZE:
            raise RuntimeError(f"풀 크기 {len(pool)} != {POOL_SIZE}")
        if CONTENT_SHA256 is not None and content_sha256(pool) != CONTENT_SHA256:
            raise RuntimeError("풀 내용 해시가 고정값과 다릅니다")
    return pool


def compare(other: Path) -> dict:
    """다시 생성한 풀과 복사본을 내용으로 대조한다."""
    a = load_pool()
    b = pd.read_parquet(other)
    sa, sb = set(a["hash"]), set(b["hash"])
    return {
        "rows": (len(a), len(b)),
        "content_equal": content_sha256(a) == content_sha256(b),
        "hash_only_in_copy": len(sa - sb), "hash_only_in_regen": len(sb - sa),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", type=Path, default=None)
    args = parser.parse_args()
    pool = load_pool()
    print({"rows": len(pool), "file_sha256_ok": True, "content_sha256": content_sha256(pool)})
    if args.compare:
        print(compare(args.compare))


if __name__ == "__main__":
    main()
