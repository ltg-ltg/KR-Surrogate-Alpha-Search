"""Study 1 연결: submodule 고정 커밋, 문법이 설계서 7.1절과 같은지, 연산자를 가져올 수 있는지."""
import pandas as pd
import pytest

from krsas import pool
from krsas.study1 import STUDY1_COMMIT, ensure


def test_study1_pinned_and_grammar_matches_design():
    ensure()
    from src.common.config import load_grammar_config
    g = load_grammar_config()
    assert list(g["fields"]) == ["open", "high", "low", "close", "volume", "returns"]
    assert len(g["operators"]) == 16
    assert g["windows"] == [5, 10, 20, 60, 120]
    assert g["size"] == {"min_depth": 2, "max_depth": 5, "max_nodes": 15, "max_cumulative_lookback": 250}
    assert STUDY1_COMMIT == "d2338b4"


def test_content_hash_ignores_row_order():
    df = pd.DataFrame({"formula_id": [1, 0], "formula": ["b", "a"], "hash": ["h1", "h0"]})
    assert pool.content_sha256(df) == pool.content_sha256(df.iloc[::-1])
    assert pool.content_sha256(df) != pool.content_sha256(df.assign(formula=["b", "x"]))


@pytest.mark.skipif(not pool.pool_path().exists(), reason="데이터 폴더에 pool_v1이 없음")
def test_pool_copy_matches_pinned_hashes():
    assert len(pool.load_pool()) == pool.POOL_SIZE
