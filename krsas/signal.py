"""수식 → (T, N) 시그널 (Study 1 연산자 커널을 그대로 쓰고, 시점 기준 척도와 유니버스 횡단면을 더한다).

Study 1 `src.backtest.operators.apply_op`(v1.0-results)로 연산을 계산한다. 이 모듈이 바꾸는 것은 두 가지다.

1 시점 기준 척도 (설계서 6.1절)
  패널 가격은 앞에서부터 누적한 수정계수 F로 수정한 값이다 (가격 × F, 거래량 ÷ F). 미래 사건은 들어가지 않지만
  수준은 실제 가격이 아니다 (예: 50대 1 분할 뒤 삼성전자 가격이 분할 전 척도). 시계열 연산은 창 안에서 척도가
  일정하므로 그대로 쓴다. 횡단면 비교(rank, zscore, 최종 순위)는 그날의 실제 척도로 되돌린 뒤 한다.
  모든 노드의 단위는 가격^a × 거래량^b 꼴이다 (Study 1 단위 규칙: 같은 단위끼리만 더하고 빼며, 곱·나눗셈은 지수를
  더하고 뺀다, 정규화 연산은 무차원). 앞쪽 수정값 = 실제값 × F^(a − b) 이므로 실제값 = 수정값 × F^(b − a).
  그래서 t일 시그널은 t일까지 알려진 기업 행위만으로 정해지고, 이 값은 t일 기준으로 과거 전체를 다시 수정한
  가격으로 계산한 값과 같다.

2 횡단면 = 그날의 유니버스 구성 종목 (설계서 8.1절 "유니버스 안에서 시그널 순위")
  rank·zscore의 입력에서 구성 종목이 아닌 칸을 NaN으로 둔다. 출력도 구성 종목 칸에만 값이 있다
  (구성 종목이 아니던 날의 횡단면 값은 정의하지 않는다). 그래서 이런 값 위에 시계열 연산을 걸면, 새로 편입된
  종목은 창이 다 찰 때까지 시그널이 없다 `[단계 A 확정 필요]`
"""
from __future__ import annotations

import numpy as np

from . import study1

study1.ensure()
from src.backtest.operators import apply_op  # noqa: E402  (Study 1 v1.0-results)
from src.generator.grammar import Node, load_grammar, parse  # noqa: E402

from .panel import Panel  # noqa: E402

_G = load_grammar()
LEAF_EXP = {"price": (1, 0), "volume": (0, 1), "dimensionless": (0, 0)}
CS_OPS = {"rank", "zscore"}


def exponents(node: Node) -> tuple[int, int]:
    """노드 단위의 (가격 지수, 거래량 지수)."""
    if node.is_leaf:
        return LEAF_EXP[_G.fields[node.op]]
    rule = _G.ops[node.op]["unit"]
    kids = [exponents(c) for c in node.children]
    if rule == "same" or rule == "same_binary":
        return kids[0]
    if rule == "dimless":
        return (0, 0)
    if rule == "mixed":
        (a1, b1), (a2, b2) = kids
        return (a1 + a2, b1 + b2) if node.op == "mul" else (a1 - a2, b1 - b2)
    raise ValueError(f"unknown unit rule {rule!r}")


def to_actual(x: np.ndarray, exp: tuple[int, int], scale: np.ndarray) -> np.ndarray:
    k = exp[1] - exp[0]
    if k == 0:
        return x
    with np.errstate(invalid="ignore", over="ignore"):
        return x * scale ** k


def evaluate(node: Node | str, panel: Panel, member: np.ndarray, cache: dict | None = None) -> np.ndarray:
    """앞쪽 수정 척도의 (T, N) 값. 최종 순위에는 signal()을 쓴다."""
    node = parse(node) if isinstance(node, str) else node
    cache = {} if cache is None else cache
    key = node.text
    if key in cache:
        return cache[key]
    if node.is_leaf:
        out = panel.fields[node.op]
    elif node.op in CS_OPS:
        child = node.children[0]
        x = to_actual(evaluate(child, panel, member, cache), exponents(child), panel.scale)
        x = np.where(member, x, np.nan)
        out = apply_op(node.op, [x], None)
        out[~member] = np.nan
    else:
        out = apply_op(node.op, [evaluate(c, panel, member, cache) for c in node.children], node.window)
    cache[key] = out
    return out


def signal(formula: Node | str, panel: Panel, member: np.ndarray) -> np.ndarray:
    """그날 실제 척도의 시그널. 구성 종목이 아닌 칸은 NaN. 횡단면 순위에 바로 쓴다."""
    node = parse(formula) if isinstance(formula, str) else formula
    x = to_actual(evaluate(node, panel, member), exponents(node), panel.scale)
    out = np.where(member & np.isfinite(x), x, np.nan)
    return out
