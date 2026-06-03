"""Adapter exposing the safe expression evaluator as an `AgentTool`."""

from __future__ import annotations

from local_llm.domain.tool_result import ToolRunRequest, ToolRunResult
from local_llm.domain.tool_spec import ToolExamples, ToolInputSchema, ToolSpec
from local_llm.domain.values.text_values import ToolName, ToolOutput
from local_llm.infrastructure.tools.calculator.expression_evaluator import (
    ExpressionEvaluator,
)


class CalculatorTool:
    """Evaluates safe arithmetic and reports failures as observations.

    Example:
        result = CalculatorTool().run(ToolRunRequest(name, ToolInput("2 + 2")))
    """

    def __init__(self) -> None:
        self._evaluator = ExpressionEvaluator()

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            ToolName("calculator"),
            "Evaluate safe arithmetic and numeric expressions.",
            ToolInputSchema(
                "A numeric expression using +, -, *, /, //, %, **",
                ToolExamples(("144 / 12", "2 ** 8", "(10 + 5) * 3")),
            ),
        )

    def run(self, request: ToolRunRequest) -> ToolRunResult:
        try:
            result = self._evaluator.evaluate(request.raw_input.value)
            output = ToolOutput(result.formatted())
            return ToolRunResult(request.tool_name, output, succeeded=True)
        except ValueError as error:
            return ToolRunResult(
                request.tool_name,
                ToolOutput(f"Calculator error: {error}"),
                succeeded=False,
            )
