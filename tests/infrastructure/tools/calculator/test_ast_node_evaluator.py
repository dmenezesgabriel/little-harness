"""Direct tests for the AST strategy evaluator and its exact error messages."""

from __future__ import annotations

import ast

import pytest

from local_llm.infrastructure.tools.calculator.ast_node_evaluator import (
    ExpressionTreeEvaluator,
)


def evaluate(expression: str) -> float:
    node = ast.parse(expression, mode="eval").body
    return ExpressionTreeEvaluator().evaluate(node).value


class TestExpressionTreeEvaluator:
    @pytest.mark.parametrize(
        ("expression", "expected"),
        [
            ("42", 42.0),
            ("2 + 3", 5.0),
            ("-5", -5.0),
        ],
    )
    def test_evaluates_each_node_kind(
        self,
        expression: str,
        expected: float,
    ) -> None:
        assert evaluate(expression) == expected

    def test_rejects_an_unsupported_node_naming_the_node_type(self) -> None:
        with pytest.raises(ValueError) as err:
            evaluate("abs(1)")
        assert str(err.value) == (
            "Unsupported expression node: Call. Expected a safe numeric expression."
        )

    def test_rejects_an_unsupported_binary_operator_naming_the_operator(self) -> None:
        with pytest.raises(ValueError) as err:
            evaluate("1 << 2")
        assert str(err.value) == (
            "Unsupported binary operator: LShift. "
            "Expected one of +, -, *, /, //, %, **."
        )

    def test_rejects_an_unsupported_unary_operator_naming_the_operator(self) -> None:
        with pytest.raises(ValueError) as err:
            evaluate("~1")
        assert str(err.value) == "Unsupported unary operator: Invert. Expected + or -."
