"""유니버스 규칙과 재개일 분류 (krsas.universe, krsas.validation.resume_windows). 작은 합성 데이터."""
import numpy as np
import pandas as pd
import pytest

from krsas import universe as uni
from krsas.validation import resume_windows


def test_rebalance_dates_first_trading_day_and_previous_base():
    dates = pd.to_datetime(["2010-11-29", "2010-11-30", "2010-12-01", "2010-12-02",
                            "2011-05-31", "2011-06-01", "2011-06-02", "2011-11-30", "2011-12-02"])
    got = uni.rebalance_dates(dates)
    assert got == [(pd.Timestamp("2010-12-01"), pd.Timestamp("2010-11-30")),
                   (pd.Timestamp("2011-06-01"), pd.Timestamp("2011-05-31")),
                   (pd.Timestamp("2011-12-02"), pd.Timestamp("2011-11-30"))]


def _day(rows):
    cols = ["ticker", "name", "market", "sect", "stock_type", "secugrp", "volume", "val20", "mktcap"]
    return pd.DataFrame(rows, columns=cols)


def test_exclusion_uses_point_in_time_name_and_not_reit_pattern():
    day = _day([
        ("000060", "메리츠화재", "KOSPI", "", "보통주", "주권", 1, 1, 1),            # '리츠'가 들어가도 보통주
        ("042670", "두산인프라코어", "KOSPI", "", "보통주", "주권", 1, 1, 1),
        ("100000", "가나스팩1호", "KOSDAQ", "SPAC(소속부없음)", "보통주", "주권", 1, 1, 1),
        ("100010", "다라제2호SPAC", "KOSDAQ", "", "보통주", "주권", 1, 1, 1),
        ("100020", "합병된회사", "KOSDAQ", "중견기업부", "보통주", "주권", 1, 1, 1),  # 예전 이름이 스팩이어도 오늘 이름으로 판정
        ("000075", "삼양사우", "KOSPI", "", "구형우선주", "주권", 1, 1, 1),
        ("00088K", "한화3우B", "KOSPI", "", "신형우선주", "주권", 1, 1, 1),
        ("330590", "롯데리츠", "KOSPI", "", "보통주", "부동산투자회사", 1, 1, 1),
    ])
    out = uni.exclusion(day).set_index("ticker")
    assert out.loc[["000060", "042670", "100020"], "eligible"].all()
    assert out.loc[["100000", "100010"], "spac_out"].all()
    assert out.loc[["000075", "00088K", "330590"], "rule1_out"].all()


def test_select_filters_and_admin_only_where_sect_known():
    rows = []
    for i in range(10):          # 코스피 10개: 소속부 없음 → 관리종목 조건을 쓰지 않는다
        rows.append((f"1{i:04d}0", f"P{i}", "KOSPI", "", "보통주", "주권", 100, 100 + i, 1000 - i))
    for i in range(10):          # 코스닥 10개: 소속부 있음, 하나는 관리종목
        sect = uni.ADMIN_SECT if i == 0 else "중견기업부"
        rows.append((f"2{i:04d}0", f"Q{i}", "KOSDAQ", sect, "보통주", "주권", 100, 200 + i, 500 - i))
    rows[1] = rows[1][:6] + (0,) + rows[1][7:]          # 코스피 하나 거래정지
    day = _day(rows)
    members, s = uni.select(day)
    main = members[members["universe"] == "main"]
    assert s["eligible"] == 20 and s["admin_out"] == 1 and s["halt_out"] == 1
    assert s["liq_out"] == 4                             # 하위 20% (val20 100~103)
    assert "200000" not in set(main["ticker"]) and "100010" not in set(main["ticker"])
    assert s["admin_known_kospi"] is False and s["admin_known_kosdaq"] is True
    assert set(main["bucket"]) == {"top", "middle", "bottom"}


def _resume_frame():
    # A: 3일 거래정지 중 10대 1 감자(기준가 0.1배 조정) 뒤 재개. 재개일 기준가가 수정 직전 종가의 절반
    #    (재평가) + 출자전환으로 주식수도 늘었다 → 주식수로 설명 안 됨
    # B: 거래정지 뒤 50대 1 분할로 재개, 주식수 50배 → 주식수로 설명됨
    # C: 주식수 변화 없이 재개일 기준가만 재설정 → reset_shares_same
    d = pd.date_range("2020-01-01", periods=6, freq="D")
    rows = []
    a = [(1000, 1000, 100, 1.0), (1000, 1000, 0, 1.0), (10000, 10000, 0, 0.1), (10000, 10000, 0, 1.0),
         (4800, 5000, 50, 2.0), (4900, 4800, 50, 1.0)]
    sh_a = [100, 100, 10, 10, 40, 40]
    b = [(5000, 5000, 10, 1.0), (5000, 5000, 0, 1.0), (5000, 5000, 0, 1.0), (5000, 5000, 0, 1.0),
         (110, 100, 10, 50.0), (110, 110, 10, 1.0)]
    sh_b = [10, 10, 10, 10, 500, 500]
    c = [(800, 800, 10, 1.0), (800, 800, 0, 1.0), (800, 800, 0, 1.0), (800, 800, 0, 1.0),
         (500, 400, 10, 2.0), (500, 500, 10, 1.0)]
    sh_c = [7] * 6
    for t, seq, sh in (("A", a, sh_a), ("B", b, sh_b), ("C", c, sh_c)):
        prev = None
        for i, ((close, base, vol, ratio), s) in enumerate(zip(seq, sh)):
            rows.append(dict(date=d[i], ticker=t, name=t, market="KOSPI", common=True, close=close, base=base,
                             volume=vol, ratio=ratio, shares=s, parval=np.nan,
                             resume=(ratio != 1.0) and prev == 0 and vol > 0, listing=i == 0))
            prev = vol
    return pd.DataFrame(rows)


def test_resume_windows_classifies_against_share_change():
    ev = resume_windows(_resume_frame()).set_index("ticker")
    assert set(ev.index) == {"A", "B", "C"}               # A의 정지 중 0.1배 조정일은 resume이 아니다
    assert ev.loc["A", "adj"] == pytest.approx(0.2) and ev.loc["A", "k_sh"] == pytest.approx(0.4)
    assert ev.loc["A", "cls"] == "shares_dont_explain" and ev.loc["A", "halt_days"] == 3
    assert ev.loc["B", "cls"] == "shares_explain" and ev.loc["B", "adj"] == pytest.approx(50.0)
    assert ev.loc["C", "cls"] == "reset_shares_same" and ev.loc["C", "dev"] == pytest.approx(np.log(2.0))
