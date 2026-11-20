"""OOS 잠금 (설계서 13절): OOS 파일은 krsas.oos만 읽는다. 쓰기는 krsas.stage_b만 한다."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "krsas"
WRITER, READER = "stage_b.py", "oos.py"
PATTERN = re.compile(r"oos_locked|oos_dir\(")


def test_only_writer_and_reader_touch_oos_files():
    offenders = []
    for f in ROOT.rglob("*.py"):
        if f.name in (WRITER, READER):
            continue
        if PATTERN.search(f.read_text(encoding="utf-8")):
            offenders.append(str(f.relative_to(ROOT)))
    assert offenders == [], f"OOS 잠금 폴더를 참조하는 모듈: {offenders}"


def test_stage_b_reader_functions_do_not_read_oos():
    src = (ROOT / WRITER).read_text(encoding="utf-8")
    reader_part = src[src.index("# ---------------------------------------------------------------- 읽기"):]
    assert not PATTERN.search(reader_part)
