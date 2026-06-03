"""Parses an arithmetic expression and evaluates it to a `Number`."""

from __future__ import annotations

import ast

from local_llm.domain.values.number import Number
from local_llm.infrastructure.tools.calculator.ast_node_evaluator import (
    ExpressionTreeEvaluator,
)


class ExpressionEvaluator:
    """Safely evaluates a numeric expression string.

    Example:
        result = ExpressionEvaluator().evaluate("144 / 12")  # Number(12.0)
    """

    def __init__(self) -> None:
        self._tree_evaluator = ExpressionTreeEvaluator()

    def evaluate(self, expression: str) -> Number:
        tree = parse_expression(expression)
        return self._tree_evaluator.evaluate(tree.body)


def parse_expression(expression: str) -> ast.Expression:
    try:
        return ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise ValueError(
            f"Invalid expression: {expression}. Expected a numeric expression."
        ) from error
