"""재설정일 처리 (krsas.ingest.reset). 작은 합성 데이터."""
import numpy as np
import pandas as pd
import pytest

from krsas.ingest.reset import build

D = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"])


def _frames(rows, info_rows, rights_rows):
    """rows: (date_idx, ticker, close, base, volume, shares). 기본 공식 열은 여기서 만든다."""
    p = pd.DataFrame(rows, columns=["i", "ticker", "close", "base", "volume", "shares"])
    p["date"] = D[p["i"]]
    p = p.sort_values(["ticker", "date"])
    p["prev_close"] = p.groupby("ticker")["close"].shift(1)
    p["ratio"] = np.where(p["prev_close"].notna(), p["prev_close"] / p["base"], 1.0)
    p["r_basic"] = np.where(p["prev_close"].notna(), p["close"] / p["base"] - 1, np.nan)
    p["adj_factor"] = p.groupby("ticker")["ratio"].cumprod()
    prev_vol = p.groupby("ticker")["volume"].shift(1)
    p["listing"] = p["prev_close"].isna()
    p["resume"] = (p["ratio"] != 1) & prev_vol.eq(0) & p["volume"].gt(0)
    p["halt_adjust"] = False
    p["gap"] = False
    info = pd.DataFrame(info_rows, columns=["i", "ticker", "parval"])
    info["date"] = D[info["i"]]
    rights = pd.DataFrame(rights_rows, columns=["ticker", "action", "i"])
    for c in ["bas_dt", "ex_date", "list_date", "issue_date", "deliver_date", "merge_date"]:
        rights[c] = D[rights["i"]] if c == "bas_dt" else pd.NaT
    return p, info, rights


def _events(rows):
    e = pd.DataFrame(rows, columns=["event_date", "ticker", "name", "kind", "treatment", "components", "source", "note"])
    e["event_date"] = pd.to_datetime(e["event_date"])
    return e


def _case():
    rows = [
        # S: 인적분할. 1000 → 정지 → 존속 주식 60%(600주), 신설 N 400주 같은 날 상장
        (0, "S", 1000, 1000, 10, 1000), (1, "S", 1000, 1000, 0, 1000), (2, "S", 700, 650, 10, 600),
        (2, "N", 500, 500, 10, 400),
        # R: 3대 1 균등감자. 300 → 정지 → 주식수 1/3, 재개 종가 1000 (기준가 990)
        (0, "R", 300, 300, 10, 900), (1, "R", 300, 300, 0, 900), (2, "R", 1000, 990, 10, 300),
        # C: 유상소각 50% (1주당 200원) — 표로 처리
        (0, "C", 400, 400, 10, 100), (1, "C", 400, 400, 0, 100), (2, "C", 420, 380, 10, 50),
        # U: 표에 없는 감자 → r_basic
        (0, "U", 100, 100, 10, 100), (1, "U", 100, 100, 0, 100), (2, "U", 220, 200, 10, 40),
        # Z: 기업 행위 없이 재평가 (k = 1)
        (0, "Z", 1000, 1000, 10, 50), (1, "Z", 1000, 1000, 0, 50), (2, "Z", 450, 500, 10, 50),
    ]
    info = [(i, t, 500.0) for i, t, *_ in rows]
    rights = [("S", "회사분할", 1), ("R", "자본감소", 1), ("C", "자본감소", 1), ("U", "자본감소", 1)]
    events = _events([
        ("2020-01-06", "S", "S", "spinoff", "total_return", "self:auto;N:auto", "", ""),
        ("2020-01-06", "R", "R", "reduction", "total_return", "self:auto", "", ""),
        ("2020-01-06", "C", "C", "reduction", "total_return", "self:auto;cash:100", "", ""),
    ])
    return _frames(rows, info, rights), events


def test_reset_rules():
    (p, info, rights), events = _case()
    prices = p[["date", "ticker", "close", "volume", "shares"]]
    out, ev = build(p, prices, info, rights, events, write=False)
    e = ev.set_index("ticker")
    assert e.loc["S", "reset_kind"] == "spinoff"
    assert e.loc["S", "r_adj"] == pytest.approx((0.6 * 700 + 0.4 * 500) / 1000 - 1)
    assert e.loc["R", "r_adj"] == pytest.approx(1000 / 3 / 300 - 1)
    assert e.loc["C", "r_adj"] == pytest.approx((0.5 * 420 + 100) / 400 - 1)
    assert e.loc["U", "r_adj"] == pytest.approx(220 / 200 - 1) and not e.loc["U", "reviewed"]
    assert e.loc["Z", "reset_kind"] == "reset" and e.loc["Z", "r_adj"] == pytest.approx(450 / 1000 - 1)
    # 수정 가격 = 종가 × adj_factor_adj 의 일별 변화가 r_adj와 같다
    o = out.sort_values(["ticker", "date"])
    ap = o["close"] * o["adj_factor_adj"]
    chg = ap / ap.groupby(o["ticker"]).shift(1) - 1
    m = chg.notna()
    assert np.allclose(chg[m], o.loc[m, "r_adj"])


def test_missing_spinoff_row_stops():
    (p, info, rights), events = _case()
    prices = p[["date", "ticker", "close", "volume", "shares"]]
    with pytest.raises(KeyError):
        build(p, prices, info, rights, events[events["ticker"] != "S"], write=False)
