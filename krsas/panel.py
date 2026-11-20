"""백테스트 패널: 날짜 × 종목 배열 (설계서 5.5절 수정 수익률, 6.1절 시점 기준, 8.1절 체결 규칙).

    python -m krsas.panel          # {data_root}/panel/panel_v1.npz 로 캐시

열은 주 유니버스나 민감도 유니버스에 한 번이라도 든 종목이다. 횡단면 연산은 그날 유니버스 구성 종목만
쓰므로(krsas.signal), 나중에 편입되는 종목이 열에 있어도 그 전 날짜의 횡단면에는 들어가지 않는다.

시그널 필드 (Study 1 문법의 필드 6개: open, high, low, close, volume, returns)
    가격 = 원주가 × adj_factor_adj (앞에서부터 누적한 수정계수. t까지의 기업 행위만 반영, 6.1절)
    거래량 = 원 거래량 / adj_factor_adj (분할 뒤에도 같은 척도)
    returns = r_adj (5.5절 처리 뒤 일별 수익률)
    거래량 0인 날(거래정지·무거래)은 모든 필드 NaN (Study 1 규칙과 같다. 시가·고가·저가가 0으로 오기 때문)
    scale = adj_factor_adj. 횡단면 연산 직전에 단위 지수로 그날의 실제 가격 척도로 되돌릴 때 쓴다

포트폴리오 배열
    ret       보유 수익률: r_adj, 거래정지일 0, 상장 전·상장폐지 뒤 NaN
    tradable  거래량 > 0 (체결 가능)
    last_row  종목의 마지막 거래일 행. 그 날이 패널 마지막 날보다 앞이면 상장폐지 청산일 (5.2절)
    kosdaq    그날 시장이 코스닥인지 (거래세율)
    member_main / member_sens   시그널일 기준 유니버스 구성 (재구성일 eff부터 다음 eff 전날까지)
    bucket_main                 주 유니버스 슬리피지 구간 0 상위, 1 중간, 2 하위 (구성 아닌 날 -1)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .ingest.clean import clean_dir
from .paths import data_root

FIELDS = ("open", "high", "low", "close", "volume", "returns")
BUCKETS = {"top": 0, "middle": 1, "bottom": 2}
VERSION = 1


@dataclass
class Panel:
    dates: pd.DatetimeIndex
    tickers: np.ndarray
    fields: dict[str, np.ndarray]       # (T, N) float64
    scale: np.ndarray                   # (T, N) adj_factor_adj, 상장 기간 밖 NaN
    valid: np.ndarray                   # (T, N) bool: 거래량 > 0
    ret: np.ndarray
    tradable: np.ndarray
    last_row: np.ndarray                # (N,) int
    kosdaq: np.ndarray                  # (T, N) bool
    member_main: np.ndarray
    member_sens: np.ndarray
    bucket_main: np.ndarray             # (T, N) int8

    def rows(self, start: str, end: str) -> slice:
        lo = self.dates.searchsorted(pd.Timestamp(start), side="left")
        hi = self.dates.searchsorted(pd.Timestamp(end), side="right")
        return slice(int(lo), int(hi))

    def truncated(self, n_rows: int) -> "Panel":
        """앞의 n_rows 날짜만 (절단 테스트, 6.1절). last_row는 자르지 않는다 (청산 판정은 미래 정보 허용, 6.1절)."""
        cut = lambda a: a[:n_rows]
        return Panel(self.dates[:n_rows], self.tickers, {k: cut(v) for k, v in self.fields.items()},
                     cut(self.scale), cut(self.valid), cut(self.ret), cut(self.tradable), self.last_row,
                     cut(self.kosdaq), cut(self.member_main), cut(self.member_sens), cut(self.bucket_main))


def _pivot(df: pd.DataFrame, col: str, dates, tickers, fill=np.nan, dtype=np.float64) -> np.ndarray:
    return df.pivot(index="date", columns="ticker", values=col).reindex(index=dates, columns=tickers) \
        .to_numpy(dtype=dtype, na_value=fill)


def build(members: pd.DataFrame | None = None) -> Panel:
    root = clean_dir()
    members = pd.read_parquet(root / "universe.parquet") if members is None else members
    tickers = np.array(sorted(members["ticker"].unique()))
    prices = pd.read_parquet(root / "prices", columns=["date", "ticker", "market", "open", "high", "low", "close",
                                                       "volume"], filters=[("ticker", "in", list(tickers))])
    rets = pd.read_parquet(root / "returns.parquet", columns=["date", "ticker", "r_adj", "adj_factor_adj"],
                           filters=[("ticker", "in", list(tickers))])
    df = prices.merge(rets, on=["date", "ticker"])
    dates = pd.DatetimeIndex(sorted(pd.read_parquet(root / "prices", columns=["date"])["date"].unique()))

    vol = _pivot(df, "volume", dates, tickers)
    listed = np.isfinite(vol)
    valid = listed & (vol > 0)
    scale = _pivot(df, "adj_factor_adj", dates, tickers)
    fields = {}
    for f in ("open", "high", "low", "close"):
        fields[f] = np.where(valid, _pivot(df, f, dates, tickers) * scale, np.nan)
    fields["volume"] = np.where(valid, vol / scale, np.nan)
    r = _pivot(df, "r_adj", dates, tickers)
    fields["returns"] = np.where(valid, r, np.nan)
    ret = np.where(listed, np.nan_to_num(r, nan=0.0), np.nan)      # 상장 첫날 r_adj NaN → 0 (보유 불가한 날)
    last_row = np.where(listed.any(axis=0), listed.shape[0] - 1 - np.argmax(listed[::-1], axis=0), -1)
    kosdaq = _pivot(df.assign(k=(df["market"] == "KOSDAQ").astype(float)), "k", dates, tickers, fill=0.0) > 0.5

    member_main = np.zeros((len(dates), len(tickers)), dtype=bool)
    member_sens = np.zeros_like(member_main)
    bucket = np.full(member_main.shape, -1, dtype=np.int8)
    col = {t: j for j, t in enumerate(tickers)}
    effs = sorted(members["eff"].unique())
    bounds = [dates.searchsorted(e) for e in effs] + [len(dates)]
    for k, e in enumerate(effs):
        lo, hi = bounds[k], bounds[k + 1]
        m = members[members["eff"] == e]
        for u, arr in (("main", member_main), ("sens", member_sens)):
            idx = [col[t] for t in m.loc[m["universe"] == u, "ticker"]]
            arr[lo:hi, idx] = True
        mm = m[m["universe"] == "main"]
        bucket[lo:hi, [col[t] for t in mm["ticker"]]] = mm["bucket"].map(BUCKETS).to_numpy(dtype=np.int8)
    return Panel(dates, tickers, fields, scale, valid, ret, valid.copy(), last_row.astype(np.int64), kosdaq,
                 member_main, member_sens, bucket)


def cache_path() -> Path:
    return data_root() / "panel" / f"panel_v{VERSION}.npz"


def save(p: Panel, path: Path | None = None) -> Path:
    path = path or cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, dates=p.dates.values.astype("datetime64[ns]"), tickers=p.tickers, scale=p.scale, valid=p.valid,
             ret=p.ret, tradable=p.tradable, last_row=p.last_row, kosdaq=p.kosdaq, member_main=p.member_main,
             member_sens=p.member_sens, bucket_main=p.bucket_main, **{f"f_{k}": v for k, v in p.fields.items()})
    return path


def load(path: Path | None = None) -> Panel:
    z = np.load(path or cache_path(), allow_pickle=False)
    return Panel(pd.DatetimeIndex(z["dates"]), z["tickers"], {f: z[f"f_{f}"] for f in FIELDS}, z["scale"],
                 z["valid"], z["ret"], z["tradable"], z["last_row"], z["kosdaq"], z["member_main"],
                 z["member_sens"], z["bucket_main"])


def npy_dir() -> Path:
    return data_root() / "panel" / f"panel_v{VERSION}_npy"


def save_npy(p: Panel, path: Path | None = None) -> Path:
    """작업자들이 mmap으로 같은 메모리를 나눠 쓰도록 배열마다 .npy로 저장한다."""
    path = path or npy_dir()
    path.mkdir(parents=True, exist_ok=True)
    arrays = {"dates": p.dates.values.astype("datetime64[ns]"), "tickers": p.tickers, "scale": p.scale,
              "valid": p.valid, "ret": p.ret, "tradable": p.tradable, "last_row": p.last_row, "kosdaq": p.kosdaq,
              "member_main": p.member_main, "member_sens": p.member_sens, "bucket_main": p.bucket_main,
              **{f"f_{k}": v for k, v in p.fields.items()}}
    for k, v in arrays.items():
        np.save(path / f"{k}.npy", np.ascontiguousarray(v))
    return path


def load_mmap(path: Path | None = None) -> Panel:
    """save_npy로 저장한 패널을 읽기 전용 mmap으로 연다."""
    path = path or npy_dir()
    m = lambda k: np.load(path / f"{k}.npy", mmap_mode="r", allow_pickle=False)
    return Panel(pd.DatetimeIndex(np.load(path / "dates.npy")), np.load(path / "tickers.npy"),
                 {f: m(f"f_{f}") for f in FIELDS}, m("scale"), m("valid"), m("ret"), m("tradable"),
                 np.load(path / "last_row.npy"), m("kosdaq"), m("member_main"), m("member_sens"), m("bucket_main"))


def main() -> None:
    p = build()
    path = save(p)
    save_npy(p)
    print(f"패널 {len(p.dates)}일 × {len(p.tickers)}종목 → {path}")
    print("주 유니버스 구성 수 범위", int(p.member_main.sum(1)[p.member_main.any(1)].min()),
          int(p.member_main.sum(1).max()), "/ 상장폐지 청산 종목", int((p.last_row < len(p.dates) - 1).sum()))


if __name__ == "__main__":
    main()
