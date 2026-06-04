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
    render_tool_spec,
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

    def test_joins_multiple_tool_specs_with_a_single_newline(self) -> None:
        # Arrange
        first = tool_spec("alpha")
        second = tool_spec("beta")

        # Act
        prompt = render_system_prompt([first, second]).value

        # Assert: the two specs are separated by exactly one newline.
        block = f"{render_tool_spec(first)}\n{render_tool_spec(second)}"
        assert block in prompt


class TestRenderToolSpec:
    def test_renders_description_input_and_joined_examples(self) -> None:
        # Act
        rendered = render_tool_spec(tool_spec("calc", ("2 + 2", "3 + 3")))

        # Assert: exact text, examples comma-joined.
        assert rendered == (
            "calc: Evaluate arithmetic. Input: A numeric expression. "
            "Examples: 2 + 2, 3 + 3."
        )

    def test_omits_the_examples_clause_when_there_are_none(self) -> None:
        # Act
        rendered = render_tool_spec(tool_spec("calc"))

        # Assert
        assert rendered == "calc: Evaluate arithmetic. Input: A numeric expression."


class TestRenderToolObservation:
    @pytest.mark.parametrize(
        ("succeeded", "status"),
        [(True, "succeeded"), (False, "failed")],
    )
    def test_renders_the_exact_observation(self, succeeded: bool, status: str) -> None:
        # Arrange
        result = ToolRunResult(ToolName("calculator"), ToolOutput("4"), succeeded)

        # Act
        message = render_tool_observation(Prompt("What is 2 + 2?"), result).value

        # Assert
        assert message == (
            "Original user question:\nWhat is 2 + 2?\n\n"
            f"Tool observation (calculator, {status}):\n4\n\n"
            "Now answer the full original user question.\n"
            "Return only one valid JSON object with action='final'."
        )


class TestRenderRepairRequest:
    def test_renders_the_exact_tool_agnostic_repair_message(self) -> None:
        # Arrange
        error = AgentProtocolError("Expected JSON object")

        # Act
        message = render_repair_request(Prompt("original question"), error).value

        # Assert
        assert message == (
            "Your previous response was invalid. Error: Expected JSON object\n\n"
            "Original user question:\noriginal question\n\n"
            "Return only one valid JSON object.\n\n"
            "Use this schema for a final answer:\n"
            '{"action":"final","tool_name":null,"tool_input":null,"answer":"..."}'
            "\n\n"
            "Use this schema for a tool call:\n"
            '{"action":"tool","tool_name":"tool_name","tool_input":"tool input",'
            '"answer":null}\n\n'
            "Do not return plain text. Do not return multiple JSON objects."
        )
        assert "calculator" not in message
