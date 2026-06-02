from __future__ import annotations

import pytest

from local_llm.agent import AgentProtocolError
from local_llm.json_policy import JsonAgentPolicy
from local_llm.tools import ToolInputSchema, ToolRunResult, ToolSpec


class TestJsonAgentPolicyParseModelOutput:
    def test_accepts_final_json_with_surrounding_text(self) -> None:
        # Arrange
        policy = JsonAgentPolicy()
        output = (
            'ignored prefix {"action":"final","tool_name":null,'
            '"tool_input":null,"answer":" done "}'
        )

        # Act
        decision = policy.parse_model_output(output)

        # Assert
        assert decision.kind == "final"
        assert decision.final_answer == "done"

    def test_rejects_missing_json_object(self) -> None:
        # Arrange
        policy = JsonAgentPolicy()
        output = "plain text"

        # Act / Assert
        with pytest.raises(AgentProtocolError, match="Could not find JSON object"):
            policy.parse_model_output(output)

    @pytest.mark.parametrize(
        ("payload", "message"),
        [
            (
                '{"action":"invalid","tool_name":null,"tool_input":null,"answer":null}',
                "Expected action",
            ),
            (
                '{"action":"final","tool_name":null,"tool_input":null,"answer":null}',
                "Expected non-null string",
            ),
            (
                '{"action":"tool","tool_name":null,"tool_input":"2 + 2","answer":null}',
                "Expected non-null string",
            ),
            (
                '{"action":"tool","tool_name":"calculator",'
                '"tool_input":null,"answer":null}',
                "Expected non-null string",
            ),
            (
                '{"action":"final","tool_name":1,"tool_input":null,"answer":"done"}',
                "Expected tool_name",
            ),
            (
                '{"action":"tool","tool_name":"calc","tool_input":2,"answer":null}',
                "Expected tool_input",
            ),
            (
                '{"action":"final","tool_name":null,"tool_input":null,"answer":3}',
                "Expected answer",
            ),
        ],
    )
    def test_rejects_invalid_decision_shape(
        self,
        payload: str,
        message: str,
    ) -> None:
        # Arrange
        policy = JsonAgentPolicy()

        # Act / Assert
        with pytest.raises(AgentProtocolError, match=message):
            policy.parse_model_output(payload)

    def test_accepts_tool_call_with_name_and_input(self) -> None:
        # Arrange
        policy = JsonAgentPolicy()
        output = (
            '{"action":"tool","tool_name":"calculator",'
            '"tool_input":"2 + 2","answer":null}'
        )

        # Act
        decision = policy.parse_model_output(output)

        # Assert
        assert decision.kind == "tool"
        assert decision.tool_name == "calculator"
        assert decision.tool_input == "2 + 2"
        assert decision.final_answer is None

    def test_extracts_first_json_object_when_answer_contains_brace(self) -> None:
        # Arrange: a stray '{' later in the text must not shift parsing.
        # (extract uses find, not rfind, so the first object wins.)
        policy = JsonAgentPolicy()
        output = (
            '{"action":"final","tool_name":null,'
            '"tool_input":null,"answer":"see {here}"}'
        )

        # Act
        decision = policy.parse_model_output(output)

        # Assert
        assert decision.final_answer == "see {here}"


class TestJsonAgentPolicyPrompts:
    def test_system_prompt_lists_tool_with_examples(self) -> None:
        # Arrange
        policy = JsonAgentPolicy()
        tool = ToolSpec(
            name="calculator",
            description="Evaluate arithmetic.",
            input_schema=ToolInputSchema("A numeric expression", ("2 + 2",)),
        )

        # Act
        prompt = policy.system_prompt([tool])

        # Assert
        assert "calculator: Evaluate arithmetic." in prompt
        assert "Input: A numeric expression." in prompt
        assert "Examples: 2 + 2." in prompt
        assert "exactly one valid JSON object" in prompt

    def test_system_prompt_omits_examples_when_absent(self) -> None:
        # Arrange
        policy = JsonAgentPolicy()
        tool = ToolSpec(
            name="echo",
            description="Echo input.",
            input_schema=ToolInputSchema("Any text"),
        )

        # Act
        prompt = policy.system_prompt([tool])

        # Assert
        assert "echo: Echo input. Input: Any text." in prompt
        assert "Examples:" not in prompt

    @pytest.mark.parametrize(
        ("succeeded", "expected_status"),
        [(True, "succeeded"), (False, "failed")],
    )
    def test_tool_observation_reports_status(
        self,
        succeeded: bool,
        expected_status: str,
    ) -> None:
        # Arrange
        policy = JsonAgentPolicy()
        result = ToolRunResult("calculator", "4", succeeded)

        # Act
        message = policy.build_tool_observation_message("What is 2 + 2?", result)

        # Assert
        assert message.role == "user"
        assert "What is 2 + 2?" in message.content
        assert f"(calculator, {expected_status})" in message.content
        assert "4" in message.content

    def test_repair_message_includes_prompt_and_error(self) -> None:
        # Arrange
        policy = JsonAgentPolicy()
        error = AgentProtocolError("Expected JSON object")

        # Act
        message = policy.build_repair_message("original question", error)

        # Assert
        assert message.role == "user"
        assert "original question" in message.content
        assert "Expected JSON object" in message.content
