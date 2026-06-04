from __future__ import annotations

from little_harness.domain.values.text_values import ToolName
from little_harness_calculator.calculator_tool import CalculatorTool
from little_harness_calculator.provider import build


class TestBuild:
    def test_returns_a_calculator_tool(self) -> None:
        # Act
        tool = build()

        # Assert
        assert isinstance(tool, CalculatorTool)
        assert tool.spec.name == ToolName("calculator")
