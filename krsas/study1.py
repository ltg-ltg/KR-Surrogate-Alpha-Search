"""Study 1 코드 연결 (설계서 7.1절, 체크리스트 8절 "공통 코드는 Study 1 태그를 지정한 의존성으로").

Study 1(`Surrogate-Alpha-Search`)은 third_party/surrogate-alpha-search에 git submodule로 두고
태그 v1.0-results(d2338b4)에 고정한다. Study 1 코드는 최상위 패키지 `src`를 절대 경로로 가져오므로
이 연구 패키지는 `krsas`로 이름을 달리하고, submodule 루트를 sys.path에 넣어 `src.*`가 Study 1을 가리키게 한다.

    from krsas.study1 import ensure
    ensure()
    from src.generator.grammar import ...        # Study 1 v1.0-results

Study 1의 config/grammar.yaml, experiment.yaml도 submodule 안의 것(고정 커밋)을 읽는다.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .paths import ROOT

STUDY1_DIR = ROOT / "third_party" / "surrogate-alpha-search"
STUDY1_TAG = "v1.0-results"
STUDY1_COMMIT = "d2338b4"


def ensure(check_commit: bool = True) -> Path:
    """submodule 루트를 sys.path 맨 앞에 넣는다. 고정 커밋이 아니면 멈춘다."""
    if not (STUDY1_DIR / "src" / "generator" / "grammar.py").exists():
        raise RuntimeError("Study 1 submodule이 없습니다: git submodule update --init")
    if check_commit:
        head = subprocess.run(["git", "-C", str(STUDY1_DIR), "rev-parse", "--short=7", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
        if head != STUDY1_COMMIT:
            raise RuntimeError(f"Study 1 submodule이 {head}입니다. {STUDY1_TAG}({STUDY1_COMMIT})로 되돌리세요")
    loaded = sys.modules.get("src")
    if loaded is not None and Path(loaded.__file__).resolve().parent != (STUDY1_DIR / "src").resolve():
        raise RuntimeError(f"다른 'src' 패키지가 이미 로드돼 있습니다: {loaded.__file__}")
    path = str(STUDY1_DIR)
    if path not in sys.path:
        sys.path.insert(0, path)
    return STUDY1_DIR
