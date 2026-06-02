from __future__ import annotations

import pytest

from local_llm.calculator import CalculatorTool, evaluate_math_expression, format_number
from local_llm.tools import ToolRunRequest, ToolRunResult


class TestEvaluateMathExpression:
    @pytest.mark.parametrize(
        ("expression", "expected"),
        [
            ("144 / 12", 12.0),
            ("2 ** 8", 256.0),
            ("(10 + 5) * 3", 45.0),
            ("7 // 2", 3.0),
            ("7 % 2", 1.0),
            ("-5 + +2", -3.0),
            # Boundary: base/exponent exactly at the limit must be allowed.
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
        result = evaluate_math_expression(expression)

        # Assert
        assert result == expected

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
            evaluate_math_expression(expression)


class TestCalculatorTool:
    def test_returns_successful_tool_result_for_valid_expression(self) -> None:
        # Arrange
        tool = CalculatorTool()

        # Act
        result = tool.run(ToolRunRequest("calculator", "2 + 2"))

        # Assert
        assert result == ToolRunResult("calculator", "4", True)

    def test_returns_failed_tool_result_for_invalid_expression(self) -> None:
        # Arrange
        tool = CalculatorTool()

        # Act
        result = tool.run(ToolRunRequest("calculator", "2 +"))

        # Assert
        assert result.tool_name == "calculator"
        assert result.succeeded is False
        assert result.output.startswith("Calculator error: Invalid expression")


class TestFormatNumber:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (4.0, "4"),
            (4.25, "4.25"),
        ],
    )
    def test_formats_integer_like_floats_without_decimal(
        self,
        value: float,
        expected: str,
    ) -> None:
        # Act
        result = format_number(value)

        # Assert
        assert result == expected
