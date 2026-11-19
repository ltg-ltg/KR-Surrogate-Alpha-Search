import gzip
import json

import pytest

from krsas import paths


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("KRSAS_DATA_ROOT", str(tmp_path / "data"))
    return tmp_path / "data"


def _write_src(root, service, bas_dd, rows=({"x": 1},)):
    p = root / service / bas_dd[:4] / f"{bas_dd}.json.gz"
    p.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(p, "wt", encoding="utf-8") as fh:
        json.dump(list(rows), fh)
    return p


def test_check_date_window():
    assert paths.check_date("20100104") == "20100104"
    assert paths.check_date("20231229") == "20231229"
    for bad in ("20240102", "20091230"):
        with pytest.raises(paths.OutOfWindow):
            paths.check_date(bad)
    with pytest.raises(ValueError):
        paths.check_date("2024-01-02")


def test_raw_path_and_fetch_refuse_reserve_period(data_root):
    from krsas.ingest.krx import KrxClient, raw_path
    with pytest.raises(paths.OutOfWindow):
        raw_path("stk_trd", "20240102")
    client = KrxClient(api_key="dummy")
    with pytest.raises(paths.OutOfWindow):
        client.fetch("stk_trd", "20240102")


def test_import_local_copies_window_only(tmp_path, data_root):
    from krsas.ingest import manifest
    from krsas.ingest.import_local import import_local
    src = tmp_path / "src"
    for svc in ("stk_trd", "ksq_trd", "stk_info", "ksq_info", "kospi_idx"):
        for d in ("20231229", "20240102"):
            _write_src(src, svc, d)
    (src / "_calendar.json").write_text(json.dumps({"20231229": True, "20240101": False, "20240102": True}))

    stats = import_local(src, "test")
    assert stats["copied"] == 4                       # 서비스 4개 × 2023년 1일. kospi_idx는 대상 아님
    assert stats["skipped_out_of_window"] == 4
    assert not list(data_root.rglob("2024*"))
    assert not (data_root / "raw" / "krx" / "kospi_idx").exists()
    cal = json.loads((data_root / "raw" / "krx" / "_calendar.json").read_text())
    assert cal == {"20231229": True}

    check = manifest.verify()
    assert check["ok"] == 4 and not check["hash_mismatch"] and not check["unlisted"] and not check["out_of_window"]
    e = next(iter(manifest.load().values()))
    assert e["via"] == "test" and e["endpoint"].startswith("https://")

    again = import_local(src, "test")                  # 두 번째 실행은 아무것도 다시 복사하지 않는다
    assert again["copied"] == 0 and again["already"] == 4


def test_import_local_refuses_to_overwrite_different_raw(tmp_path, data_root):
    from krsas.ingest.import_local import import_local
    src = tmp_path / "src"
    _write_src(src, "stk_trd", "20231229")
    import_local(src, "test")
    _write_src(src, "stk_trd", "20231229", rows=({"x": 2},))
    with pytest.raises(RuntimeError):
        import_local(src, "test")


def test_verify_flags_out_of_window_files(data_root):
    from krsas.ingest import manifest
    _write_src(data_root / "raw" / "krx", "stk_trd", "20240102")
    check = manifest.verify()
    assert check["out_of_window"] == ["stk_trd/2024/20240102.json.gz"]


class _FakeClient:
    budget = None

    def __init__(self, holidays=()):
        self.holidays = set(holidays)
        self.calls = []

    def fetch(self, service, bas_dd):
        self.calls.append((service, bas_dd))
        return [] if bas_dd in self.holidays else [{"x": 1}]


def test_collect_stays_in_window_and_records_manifest(data_root, monkeypatch):
    from krsas.ingest import collect as c
    from krsas.ingest import manifest
    monkeypatch.setattr(c, "target_days", lambda: ["20231228", "20231229"])
    client = _FakeClient(holidays={"20231229"})
    stats = c.collect(client)
    assert stats["days_done"] == 1 and stats["holidays"] == 1 and stats["files"] == 4
    assert c.load_calendar() == {"20231228": True, "20231229": False}
    assert manifest.verify()["ok"] == 4
    assert all(d <= paths.DATA_END for _, d in client.calls)


def test_target_days_bounds():
    from krsas.ingest.collect import target_days
    days = target_days()
    assert days[0] == paths.DATA_START and days[-1] <= paths.DATA_END


class _FakeRights:
    """페이지를 나눠 주는 가짜 응답."""
    budget = None

    def __init__(self, per_day):
        self.per_day = per_day
        self.calls = []

    def _page(self, bas_dd, page):
        self.calls.append((bas_dd, page))
        rows = [{"basDt": bas_dd, "i": i} for i in range(self.per_day.get(bas_dd, 0))]
        chunk = rows[(page - 1) * 2: page * 2]
        return {"totalCount": len(rows), "items": {"item": chunk} if chunk else ""}


def test_rights_collect_pages_and_window(data_root, monkeypatch):
    from krsas.ingest import rights
    monkeypatch.setattr(rights, "PAGE_SIZE", 2)
    monkeypatch.setattr(rights, "target_days", lambda: ["20231230", "20231231"])
    fake = _FakeRights({"20231231": 5})
    fake.fetch = rights.RightsClient.fetch.__get__(fake)
    stats = rights.collect(fake)
    assert stats["days_done"] == 2 and stats["rows"] == 5 and stats["remaining_days"] == 0
    assert rights.read_raw("20231230") == [] and len(rights.read_raw("20231231")) == 5
    # 0건인 날은 한 번 더 확인한다
    assert fake.calls == [("20231230", 1), ("20231230", 1), ("20231231", 1), ("20231231", 2), ("20231231", 3)]
    with pytest.raises(paths.OutOfWindow):
        fake.fetch("20240102")


def test_rights_target_days_bounds():
    from krsas.ingest import rights
    days = rights.target_days()
    assert days[0] == paths.DATA_START and days[-1] == "20231231" and "20100109" in days   # 주말 포함


def test_rights_retries_gateway_error_and_maps_limit(data_root, monkeypatch):
    from krsas.ingest import rights
    from krsas.ingest.krx import BudgetExceeded
    monkeypatch.setattr(rights.time, "sleep", lambda s: None)

    class Resp:
        status_code = 200
        encoding = None

        def __init__(self, data):
            self._d = data

        def json(self):
            return self._d

    gw = lambda code: {"OpenAPI_ServiceResponse": {"cmmMsgHeader": {"returnReasonCode": code}}}
    ok = {"response": {"header": {"resultCode": "00"}, "body": {"totalCount": 0, "items": ""}}}
    c = rights.RightsClient(api_key="dummy")
    seq = [Resp(gw("04")), Resp(ok)]
    c.session.get = lambda *a, **k: seq.pop(0)
    assert c._page("20101231", 16)["totalCount"] == 0 and not seq
    c.session.get = lambda *a, **k: Resp(gw("22"))
    with pytest.raises(BudgetExceeded):
        c._page("20101231", 1)
    c.session.get = lambda *a, **k: Resp(gw("30"))       # 등록되지 않은 키 등은 바로 멈춘다
    with pytest.raises(RuntimeError):
        c._page("20101231", 1)


def test_rights_refetches_bogus_empty_page(data_root, monkeypatch):
    """중간 페이지가 정상 코드로 빈 응답(totalCount 0)을 주면 다시 불러 이어 간다."""
    from krsas.ingest import rights
    monkeypatch.setattr(rights, "PAGE_SIZE", 2)
    monkeypatch.setattr(rights.time, "sleep", lambda s: None)
    fake = _FakeRights({"20101231": 5})
    real_page = fake._page
    bad = {("20101231", 2)}

    def flaky(bas_dd, page):
        if (bas_dd, page) in bad:
            bad.discard((bas_dd, page))
            fake.calls.append((bas_dd, page))
            return {"totalCount": 0, "items": ""}
        return real_page(bas_dd, page)

    fake._page = flaky
    fake.fetch = rights.RightsClient.fetch.__get__(fake)
    assert len(fake.fetch("20101231")) == 5
    assert fake.calls.count(("20101231", 2)) == 2
