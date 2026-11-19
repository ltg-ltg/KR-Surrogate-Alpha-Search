"""경로와 연구 기간의 경계.

데이터는 레포 밖에 둔다 (KRX 약관: 제3자 제공 금지). 기본 위치는 C:\\kr_surrogate_alpha_search_data이고
환경 변수 KRSAS_DATA_ROOT로 바꿀 수 있다.

이 연구가 읽거나 받을 수 있는 날짜는 DATA_START ~ DATA_END뿐이다. 2024년 이후는 예비 구간이라
데이터 폴더에 두지 않고, 코드도 받지 않는다 (설계서 6절).
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_START = "20100104"     # KRX Open API 제공 시작일
DATA_END = "20231231"       # 이 연구가 쓰는 마지막 날짜. 2024년 이후는 예비 구간


def data_root() -> Path:
    return Path(os.environ.get("KRSAS_DATA_ROOT", r"C:\kr_surrogate_alpha_search_data"))


def krx_raw_dir() -> Path:
    return data_root() / "raw" / "krx"


class OutOfWindow(ValueError):
    pass


def check_date(bas_dd: str) -> str:
    """YYYYMMDD가 연구 기간 안인지 확인한다. 밖이면 OutOfWindow."""
    if len(bas_dd) != 8 or not bas_dd.isdigit():
        raise ValueError(f"날짜 형식은 YYYYMMDD: {bas_dd!r}")
    if not DATA_START <= bas_dd <= DATA_END:
        raise OutOfWindow(f"{bas_dd}는 연구 기간({DATA_START}~{DATA_END}) 밖입니다")
    return bas_dd
