from __future__ import annotations

import pytest

from local_llm.domain.errors import AgentProtocolError
from local_llm.domain.tool_result import ToolRunResult
from local_llm.domain.tool_spec import ToolExamples, ToolInputSchema, ToolSpec
from local_llm.domain.values.text_values import (
    Prompt,
    ToolName,
    ToolOutput,
)
from local_llm.infrastructure.policy.prompt_templates import (
    render_repair_request,
    render_system_prompt,
    render_tool_observation,
)


def tool_spec(name: str, examples: tuple[str, ...] = ()) -> ToolSpec:
    return ToolSpec(
        ToolName(name),
        "Evaluate arithmetic.",
        ToolInputSchema("A numeric expression", ToolExamples(examples)),
    )


class TestRenderSystemPrompt:
    def test_lists_tool_with_examples(self) -> None:
        # Act
        prompt = render_system_prompt([tool_spec("calculator", ("2 + 2",))]).value

        # Assert
        assert "calculator: Evaluate arithmetic." in prompt
        assert "Input: A numeric expression." in prompt
        assert "Examples: 2 + 2." in prompt
        assert "exactly one valid JSON object" in prompt

    def test_example_call_uses_passed_tool_not_a_hardcoded_one(self) -> None:
        # Act
        prompt = render_system_prompt([tool_spec("echo", ("hello",))]).value

        # Assert
        assert '"tool_name": "echo"' in prompt
        assert '"tool_input": "hello"' in prompt
        assert "calculator" not in prompt

    def test_example_call_falls_back_when_no_tools(self) -> None:
        # Act
        prompt = render_system_prompt([]).value

        # Assert
        assert '"tool_name": "tool_name"' in prompt
        assert "calculator" not in prompt

    def test_omits_examples_when_absent(self) -> None:
        # Act
        prompt = render_system_prompt([tool_spec("echo")]).value

        # Assert
        assert "echo: Evaluate arithmetic. Input: A numeric expression." in prompt
        assert "Examples:" not in prompt


class TestRenderToolObservation:
    @pytest.mark.parametrize(
        ("succeeded", "expected_status"),
        [(True, "succeeded"), (False, "failed")],
    )
    def test_reports_status(self, succeeded: bool, expected_status: str) -> None:
        # Arrange
        result = ToolRunResult(ToolName("calculator"), ToolOutput("4"), succeeded)

        # Act
        message = render_tool_observation(Prompt("What is 2 + 2?"), result).value

        # Assert
        assert "What is 2 + 2?" in message
        assert f"(calculator, {expected_status})" in message
        assert "4" in message


class TestRenderRepairRequest:
    def test_includes_prompt_and_error_and_stays_tool_agnostic(self) -> None:
        # Arrange
        error = AgentProtocolError("Expected JSON object")

        # Act
        message = render_repair_request(Prompt("original question"), error).value

        # Assert
        assert "original question" in message
        assert "Expected JSON object" in message
        assert "calculator" not in message
