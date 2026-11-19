"""정제·수정계수: 형 변환, 기준가로 찾는 이벤트, 앞에서부터 누적하는 계수, 재설정일 후보.
kr_quant_machine tests/test_krx.py에서 가져와 재설정일 후보 검사를 더했다."""
import pandas as pd
import pytest

from src import costs
from src.ingest.adjust import build
from src.ingest.clean import INFO_COLS, TRD_COLS, to_frame


def trd(code, close, change, open_=None, vol=100):
    return {"BAS_DD": "x", "ISU_CD": code, "ISU_NM": f"N{code}", "MKT_NM": "KOSPI", "SECT_TP_NM": "",
            "TDD_CLSPRC": str(close), "CMPPREVDD_PRC": str(change), "FLUC_RT": "0",
            "TDD_OPNPRC": str(open_ if open_ is not None else close), "TDD_HGPRC": str(close),
            "TDD_LWPRC": str(close), "ACC_TRDVOL": str(vol), "ACC_TRDVAL": "1,000", "MKTCAP": "1", "LIST_SHRS": "10"}


def info(code, parval, shares):
    return {"ISU_CD": f"KR7{code}000", "ISU_SRT_CD": code, "ISU_NM": "n", "ISU_ABBRV": "n", "ISU_ENG_NM": "n",
            "LIST_DD": "20000104", "MKT_TP_NM": "KOSPI", "SECUGRP_NM": "주권", "SECT_TP_NM": "",
            "KIND_STKCERT_TP_NM": "보통주", "PARVAL": parval, "LIST_SHRS": str(shares)}


def test_to_frame_types():
    df = to_frame([trd("005930", 2650000, 0)], TRD_COLS, "20180503")
    assert df.loc[0, "date"] == pd.Timestamp("2018-05-03") and df.loc[0, "ticker"] == "005930"
    assert df.loc[0, "close"] == 2650000 and df.loc[0, "value"] == 1000      # 쉼표 제거
    inf = to_frame([info("005930", "무액면", 5)], INFO_COLS, "20180503")
    assert pd.isna(inf.loc[0, "parval"]) and inf.loc[0, "list_date"] == pd.Timestamp("2000-01-04")


def _sample():
    days = ["20180502", "20180503", "20180504", "20180508"]
    # A: 50대 1 분할 (전일 2,650,000 → 기준가 53,000). B: 이벤트 없음. C: 거래정지 뒤 10대 1 병합으로 재개.
    # D: 05-03에 처음 나타남(신규 상장). E: 05-03 응답에서 빠졌다가 05-04에 기준가가 바뀌어 다시 나타남
    rows = {
        "20180502": [trd("A", 2600000, 10000), trd("B", 1000, 10), trd("C", 500, 0), trd("E", 100, 0)],
        "20180503": [trd("A", 2650000, 50000), trd("B", 1010, 10), trd("C", 500, 0, open_=0, vol=0),
                     trd("D", 2000, 0)],
        "20180504": [trd("A", 51900, -1100), trd("B", 990, -20), trd("C", 5500, 500), trd("D", 2100, 100),
                     trd("E", 60, 10)],
        "20180508": [trd("A", 52600, 700), trd("B", 1000, 10), trd("C", 5400, -100), trd("D", 2100, 0),
                     trd("E", 60, 0)],
    }
    infos = {d: [info("A", "5000", 100), info("B", "500", 10), info("C", "500", 1000)] for d in days[:2]}
    infos |= {d: [info("A", "100", 5000), info("B", "500", 10), info("C", "5000", 100)] for d in days[2:]}
    prices = pd.concat([to_frame(rows[d], TRD_COLS, d) for d in days], ignore_index=True)
    share_info = pd.concat([to_frame(infos[d], INFO_COLS, d) for d in days], ignore_index=True)
    return prices, share_info


def test_adj_factor_from_base_price():
    prices, share_info = _sample()
    returns, events = build(prices, share_info, write=False)

    assert list(events["ticker"]) == ["A", "C", "E"] and (events["date"] == pd.Timestamp("2018-05-04")).all()
    a, c = events.iloc[0], events.iloc[1]
    assert a["ratio"] == pytest.approx(50.0) and a["kind"] == "split" and a["shares"] == 5000
    assert c["ratio"] == pytest.approx(0.1) and c["kind"] == "reverse_split"

    f = returns.pivot(index="date", columns="ticker", values="adj_factor")
    assert list(f["A"]) == pytest.approx([1, 1, 50, 50]) and (f["B"] == 1).all()
    r = returns.set_index(["date", "ticker"])["r_basic"]
    d = pd.Timestamp("2018-05-04")
    assert r[(d, "A")] == pytest.approx(51900 / 53000 - 1)          # 종가 / 기준가 − 1
    assert r[(d, "C")] == pytest.approx(5500 / 5000 - 1)
    assert r[(d, "B")] == pytest.approx(990 / 1010 - 1)


def test_reset_day_candidates():
    prices, share_info = _sample()
    returns, events = build(prices, share_info, write=False)
    flags = returns.set_index(["date", "ticker"])
    d = pd.Timestamp("2018-05-04")
    assert flags.loc[(d, "C"), "resume"] and not flags.loc[(d, "C"), "gap"]     # 정지 다음 날 기준가 재설정
    assert flags.loc[(d, "E"), "gap"] and not flags.loc[(d, "E"), "resume"]     # 응답에서 빠졌다 재등장
    assert not flags.loc[(d, "A"), "resume"] and not flags.loc[(d, "A"), "gap"]  # 일반 분할
    first = flags.loc[(pd.Timestamp("2018-05-03"), "D")]
    assert first["listing"] and pd.isna(first["r_basic"])                        # 신규 상장 첫날 제외
    assert set(events.loc[events["resume"] | events["gap"], "ticker"]) == {"C", "E"}


def test_sell_tax_steps():
    assert costs.sell_tax("kospi", "2019-05-29") == 0.0030
    assert costs.sell_tax("kospi", "2019-05-30") == 0.0025
    assert costs.sell_tax("kosdaq", "2023-12-28") == 0.0020
    with pytest.raises(ValueError):
        costs.sell_tax("kospi", "2009-12-30")
