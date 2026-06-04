from __future__ import annotations

import pytest
from little_harness_calculator.expression_evaluator import (
    ExpressionEvaluator,
)


class TestExpressionEvaluator:
    @pytest.mark.parametrize(
        ("expression", "expected"),
        [
            ("144 / 12", 12.0),
            ("2 ** 8", 256.0),
            ("(10 + 5) * 3", 45.0),
            ("7 // 2", 3.0),
            ("7 % 2", 1.0),
            ("-5 + +2", -3.0),
            ("1000000 ** 2", 1_000_000_000_000.0),
            ("2 ** 12", 4096.0),
        ],
    )
    def test_evaluates_safe_numeric_expression(
        self,
        expression: str,
        expected: float,
    ) -> None:
        # Act
        result = ExpressionEvaluator().evaluate(expression)

        # Assert
        assert result.value == expected

    @pytest.mark.parametrize(
        ("expression", "message"),
        [
            ("2 +", "Invalid expression"),
            ("True", "Unsupported boolean value"),
            ("abs(-1)", "Unsupported expression node"),
            ("[1]", "Unsupported expression node"),
            ('"1"', "Unsupported constant value"),
            ("1 << 2", "Unsupported binary operator"),
            ("~1", "Unsupported unary operator"),
            ("1000001 ** 2", "Power base too large"),
            ("2 ** 13", "Power exponent too large"),
        ],
    )
    def test_rejects_unsafe_or_invalid_expression(
        self,
        expression: str,
        message: str,
    ) -> None:
        # Act / Assert
        with pytest.raises(ValueError, match=message):
            ExpressionEvaluator().evaluate(expression)
