"""거래 비용 설정 (config/costs.yaml, 설계서 8.3절)."""
from __future__ import annotations

from functools import lru_cache

import pandas as pd
import yaml

from .paths import ROOT


@lru_cache(maxsize=1)
def load() -> dict:
    return yaml.safe_load((ROOT / "config" / "costs.yaml").read_text(encoding="utf-8"))


def sell_tax(market: str, trade_date) -> float:
    """체결일 기준 매도 거래세 합계. market은 'kospi' 또는 'kosdaq'."""
    d = pd.Timestamp(trade_date)
    rate = None
    for step in load()["sell_tax"][market]:
        if pd.Timestamp(step["from"]) <= d:
            rate = step["rate"]
    if rate is None:
        raise ValueError(f"{market} {d.date()}: 세율 구간 밖")
    return rate
