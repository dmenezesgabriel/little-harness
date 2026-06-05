from __future__ import annotations

import json

import pytest
from little_harness.domain.decision import FinalAnswer, ToolCall
from little_harness.domain.errors import AgentProtocolError
from little_harness.domain.values.text_values import MessageContent, ToolInput, ToolName
from little_harness_json_policy.decision_parser import JsonDecisionParser


def parse(output: str) -> object:
    return JsonDecisionParser().parse(MessageContent(output))


class TestParsesFinalAnswers:
    def test_accepts_final_json_with_surrounding_text_and_trims(self) -> None:
        # Arrange: prose before the object, padding inside the answer.
        output = 'ignored prefix {"action":"final","answer":" done "}'

        # Act / Assert
        assert parse(output) == FinalAnswer(MessageContent("done"))

    def test_still_accepts_the_older_nested_final_shape(self) -> None:
        # Arrange: the legacy protocol carried null tool fields alongside.
        output = '{"action":"final","tool_name":null,"tool_input":null,"answer":"hi"}'

        # Act / Assert
        assert parse(output) == FinalAnswer(MessageContent("hi"))

    def test_extracts_first_json_object_when_answer_contains_a_brace(self) -> None:
        # Arrange: a stray '{' later in the text must not shift parsing.
        output = '{"action":"final","answer":"see {here}"}'

        # Act / Assert
        assert parse(output) == FinalAnswer(MessageContent("see {here}"))


class TestParsesFlattenedToolCalls:
    def test_reads_the_action_itself_as_the_tool_name(self) -> None:
        # Act / Assert: the small-model shape, where action *is* the tool name.
        assert parse('{"action":"calculator","input":"2 + 2"}') == ToolCall(
            ToolName("calculator"), ToolInput("2 + 2")
        )

    def test_serializes_an_object_input_back_to_a_json_string(self) -> None:
        # Arrange: file tools need a JSON object, not a doubly-encoded string.
        output = '{"action":"write_file","input":{"path":"a.txt","content":"hi"}}'

        # Act
        decision = parse(output)

        # Assert: the tool receives valid JSON that round-trips to the object.
        assert isinstance(decision, ToolCall)
        assert decision.tool_name == ToolName("write_file")
        assert json.loads(decision.tool_input.value) == {
            "path": "a.txt",
            "content": "hi",
        }

    def test_prefers_input_over_a_legacy_tool_input_when_both_exist(self) -> None:
        # Act / Assert
        output = '{"action":"calculator","input":"1 + 1","tool_input":"9 + 9"}'
        assert parse(output) == ToolCall(ToolName("calculator"), ToolInput("1 + 1"))

    def test_folds_inline_top_level_arguments_into_the_input(self) -> None:
        # Arrange: the shape a local model emitted, with args beside "action".
        output = '{"action":"write_file","path":"hello.txt","content":"hello"}'

        # Act
        decision = parse(output)

        # Assert: the non-decision keys become the tool's JSON input.
        assert isinstance(decision, ToolCall)
        assert decision.tool_name == ToolName("write_file")
        assert json.loads(decision.tool_input.value) == {
            "path": "hello.txt",
            "content": "hello",
        }


class TestParsesLegacyNestedToolCalls:
    def test_reads_the_tool_name_field_for_the_tool_action(self) -> None:
        # Act / Assert
        output = '{"action":"tool","tool_name":"calculator","tool_input":"2 + 2"}'
        assert parse(output) == ToolCall(ToolName("calculator"), ToolInput("2 + 2"))

    def test_serializes_a_legacy_object_tool_input(self) -> None:
        # Act
        output = '{"action":"tool","tool_name":"write_file","tool_input":{"path":"a"}}'
        decision = parse(output)

        # Assert
        assert isinstance(decision, ToolCall)
        assert json.loads(decision.tool_input.value) == {"path": "a"}


class TestRejectsMissingOrMalformedJson:
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


class TestRejectsInvalidAction:
    @pytest.mark.parametrize(
        "payload",
        [
            '{"tool_name":"calc","tool_input":"2 + 2"}',  # action absent
            '{"action":null,"answer":"x"}',  # action null
            '{"action":"","input":"2 + 2"}',  # action empty
            '{"action":"   ","input":"2 + 2"}',  # action whitespace
            '{"action":7,"input":"2 + 2"}',  # action not a string
        ],
    )
    def test_rejects_a_non_string_or_empty_action(self, payload: str) -> None:
        # Act / Assert
        with pytest.raises(AgentProtocolError, match="Expected a tool name or 'final'"):
            parse(payload)


class TestRejectsInvalidFinalAnswer:
    @pytest.mark.parametrize(
        "payload",
        [
            '{"action":"final","answer":null}',
            '{"action":"final","answer":3}',
            '{"action":"final"}',
        ],
    )
    def test_rejects_a_missing_or_non_string_answer(self, payload: str) -> None:
        # Act / Assert
        with pytest.raises(AgentProtocolError, match="Field answer is invalid"):
            parse(payload)


class TestRejectsInvalidToolInput:
    @pytest.mark.parametrize(
        "payload",
        [
            '{"action":"calculator"}',  # no input at all
            '{"action":"calculator","input":null}',  # null input
            '{"action":"calculator","input":2}',  # non string/object input
            '{"action":"calculator","input":["a"]}',  # array is not accepted
        ],
    )
    def test_rejects_input_that_is_neither_a_string_nor_an_object(
        self, payload: str
    ) -> None:
        # Act / Assert
        with pytest.raises(AgentProtocolError, match="Tool input is invalid"):
            parse(payload)

    def test_names_the_exact_cause_when_a_tool_call_has_no_arguments(self) -> None:
        # Act / Assert: the full message guides the model on every accepted shape.
        with pytest.raises(AgentProtocolError) as err:
            parse('{"action":"calculator"}')
        assert str(err.value) == (
            "Tool input is invalid: no 'input' field and no inline arguments. "
            "Expected a JSON string, a JSON object, or arguments beside 'action'."
        )


class TestRejectsInvalidToolName:
    def test_rejects_a_missing_tool_name_in_the_legacy_shape(self) -> None:
        # Act / Assert
        with pytest.raises(AgentProtocolError, match="Field tool_name is invalid"):
            parse('{"action":"tool","tool_input":"2 + 2"}')

    def test_rejects_a_blank_tool_name_in_the_legacy_shape(self) -> None:
        # Act / Assert: a whitespace name passes the string check but is empty.
        with pytest.raises(AgentProtocolError, match="Invalid tool name in model"):
            parse('{"action":"tool","tool_name":"  ","tool_input":"2 + 2"}')
