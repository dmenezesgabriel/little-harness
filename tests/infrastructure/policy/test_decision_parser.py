from __future__ import annotations

import pytest

from local_llm.domain.decision import FinalAnswer, ToolCall
from local_llm.domain.errors import AgentProtocolError
from local_llm.domain.values.text_values import MessageContent, ToolInput, ToolName
from local_llm.infrastructure.policy.decision_parser import JsonDecisionParser


def parse(output: str) -> object:
    return JsonDecisionParser().parse(MessageContent(output))


class TestJsonDecisionParserValid:
    def test_accepts_final_json_with_surrounding_text(self) -> None:
        # Arrange
        output = (
            'ignored prefix {"action":"final","tool_name":null,'
            '"tool_input":null,"answer":" done "}'
        )

        # Act / Assert
        assert parse(output) == FinalAnswer(MessageContent("done"))

    def test_accepts_tool_call(self) -> None:
        # Arrange
        output = (
            '{"action":"tool","tool_name":"calculator",'
            '"tool_input":"2 + 2","answer":null}'
        )

        # Act / Assert
        assert parse(output) == ToolCall(ToolName("calculator"), ToolInput("2 + 2"))

    def test_extracts_first_json_object_when_answer_contains_brace(self) -> None:
        # Arrange: a stray '{' later in the text must not shift parsing.
        output = (
            '{"action":"final","tool_name":null,'
            '"tool_input":null,"answer":"see {here}"}'
        )

        # Act / Assert
        assert parse(output) == FinalAnswer(MessageContent("see {here}"))


class TestJsonDecisionParserInvalid:
    def test_rejects_missing_json_object(self) -> None:
        # Act / Assert: the full message names the cause and the guidance line.
        with pytest.raises(AgentProtocolError) as err:
            parse("plain text")
        assert str(err.value) == (
            "Could not find JSON object in model output: plain text. "
            "Expected one valid JSON object."
        )

    def test_rejects_malformed_json_after_a_brace(self) -> None:
        # Act / Assert: the full message echoes the offending text.
        with pytest.raises(AgentProtocolError) as err:
            parse("here it is { not valid json")
        assert str(err.value) == (
            "Invalid JSON object in model output: here it is { not valid json. "
            "Expected one valid JSON object."
        )


class TestJsonDecisionParserShape:
    @pytest.mark.parametrize(
        ("payload", "message"),
        [
            (
                '{"action":"invalid","tool_name":null,"tool_input":null,"answer":null}',
                "Expected action",
            ),
            (
                '{"action":"final","tool_name":null,"tool_input":null,"answer":null}',
                "Field answer is invalid",
            ),
            (
                '{"action":"final","tool_name":null,"tool_input":null,"answer":3}',
                "Field answer is invalid",
            ),
            (
                '{"action":"tool","tool_name":null,"tool_input":"2 + 2","answer":null}',
                "Field tool_name is invalid",
            ),
            (
                '{"action":"tool","tool_name":"calc","tool_input":2,"answer":null}',
                "Field tool_input is invalid",
            ),
            (
                '{"action":"tool","tool_name":"   ",'
                '"tool_input":"2 + 2","answer":null}',
                "Invalid tool name in model output",
            ),
        ],
    )
    def test_rejects_invalid_decision_shape(
        self,
        payload: str,
        message: str,
    ) -> None:
        # Act / Assert
        with pytest.raises(AgentProtocolError, match=message):
            parse(payload)
