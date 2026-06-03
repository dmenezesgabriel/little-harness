from __future__ import annotations

from local_llm.domain.tool_result import ToolRunRequest, ToolRunResult
from local_llm.domain.values.text_values import ToolInput, ToolName, ToolOutput
from local_llm.infrastructure.tools.calculator.calculator_tool import CalculatorTool


def calculator_request(raw_input: str) -> ToolRunRequest:
    return ToolRunRequest(ToolName("calculator"), ToolInput(raw_input))


class TestCalculatorTool:
    def test_advertises_its_spec(self) -> None:
        # Act
        spec = CalculatorTool().spec

        # Assert
        assert spec.name == ToolName("calculator")
        assert spec.input_schema.examples.first() == "144 / 12"

    def test_returns_successful_result_for_valid_expression(self) -> None:
        # Act
        result = CalculatorTool().run(calculator_request("2 + 2"))

        # Assert
        assert result == ToolRunResult(
            ToolName("calculator"), ToolOutput("4"), succeeded=True
        )

    def test_returns_failed_result_for_invalid_expression(self) -> None:
        # Act
        result = CalculatorTool().run(calculator_request("2 +"))

        # Assert
        assert result.tool_name == ToolName("calculator")
        assert result.succeeded is False
        assert result.output.value.startswith("Calculator error: Invalid expression")

    def test_returns_failed_result_for_division_by_zero(self) -> None:
        # Act
        result = CalculatorTool().run(calculator_request("144 / 0"))

        # Assert
        assert result.succeeded is False
        assert result.output.value.startswith("Calculator error: Division by zero")
