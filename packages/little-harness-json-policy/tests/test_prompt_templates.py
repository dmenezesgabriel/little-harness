from __future__ import annotations

import pytest
from little_harness.domain.errors import AgentProtocolError
from little_harness.domain.tool_result import ToolRunResult
from little_harness.domain.tool_spec import ToolExamples, ToolInputSchema, ToolSpec
from little_harness.domain.values.text_values import (
    Prompt,
    ToolName,
    ToolOutput,
)
from little_harness_json_policy.prompt_templates import (
    build_response_schema,
    build_tool_call_example,
    format_input_example,
    render_repair_request,
    render_system_prompt,
    render_tool_observation,
    render_tool_spec,
)


def tool_spec(
    name: str,
    examples: tuple[str, ...] = (),
    json_schema: dict[str, object] | None = None,
) -> ToolSpec:
    return ToolSpec(
        ToolName(name),
        "Evaluate arithmetic.",
        ToolInputSchema("A numeric expression", ToolExamples(examples), json_schema),
    )


class TestRenderSystemPrompt:
    def test_lists_tool_with_examples(self) -> None:
        # Act
        prompt = render_system_prompt([tool_spec("calculator", ("2 + 2",))]).value

        # Assert
        assert "calculator: Evaluate arithmetic." in prompt
        assert "Input: A numeric expression." in prompt
        assert "Examples: 2 + 2." in prompt
        assert "exactly one JSON object" in prompt

    def test_carries_no_hardcoded_calculator_final_answer(self) -> None:
        # Act: the old prompt baked a calculator answer into the generic policy.
        prompt = render_system_prompt([tool_spec("echo", ("hi",))]).value

        # Assert
        assert "144 divided by 12" not in prompt
        assert "is even" not in prompt

    def test_tool_call_example_uses_generic_placeholders(self) -> None:
        # Act
        prompt = render_system_prompt([tool_spec("echo", ("hello",))]).value

        # Assert: the example uses placeholders so small models aren't biased
        # toward the first tool. The repair message already uses this same
        # generic format.
        assert '{"action": "tool_name", "input": "tool input"}' in prompt

    def test_final_answer_format_is_shown(self) -> None:
        # Act
        prompt = render_system_prompt([tool_spec("echo")]).value

        # Assert
        assert '{"action": "final", "answer":' in prompt

    def test_joins_multiple_tool_specs_with_a_single_newline(self) -> None:
        # Arrange
        first = tool_spec("alpha")
        second = tool_spec("beta")

        # Act
        prompt = render_system_prompt([first, second]).value

        # Assert: the two specs are separated by exactly one newline.
        block = f"{render_tool_spec(first)}\n{render_tool_spec(second)}"
        assert block in prompt


class TestBuildToolCallExample:
    def test_uses_placeholders_regardless_of_tool(self) -> None:
        example = build_tool_call_example([tool_spec("edit_file", ('{"path": "a"}',))])
        assert example == '{"action": "tool_name", "input": "tool input"}'

    def test_uses_placeholders_with_bare_expressions(self) -> None:
        example = build_tool_call_example([tool_spec("calculator", ("144 / 12",))])
        assert example == '{"action": "tool_name", "input": "tool input"}'

    def test_uses_placeholders_without_tools(self) -> None:
        assert build_tool_call_example([]) == (
            '{"action": "tool_name", "input": "tool input"}'
        )

    def test_uses_placeholders_when_tool_has_no_examples(self) -> None:
        assert build_tool_call_example([tool_spec("echo")]) == (
            '{"action": "tool_name", "input": "tool input"}'
        )


class TestFormatInputExample:
    @pytest.mark.parametrize(
        ("example", "expected"),
        [
            ('{"path": "a"}', '{"path": "a"}'),  # object: inlined
            ("144 / 12", '"144 / 12"'),  # expression: quoted
            ("42", "42"),  # bare number: valid JSON, inlined
        ],
    )
    def test_inlines_json_and_quotes_everything_else(
        self, example: str, expected: str
    ) -> None:
        # Act / Assert
        assert format_input_example(example) == expected


class TestBuildResponseSchema:
    def test_builds_a_oneof_of_final_and_per_tool_branches(self) -> None:
        # Act
        schema = build_response_schema(
            [
                tool_spec("calculator", json_schema={"type": "string"}),
                tool_spec(
                    "edit_file",
                    json_schema={
                        "type": "object",
                        "required": ["path"],
                    },
                ),
            ]
        )

        # Assert: each tool's branch constrains the action and input shape.
        assert schema.value == {
            "oneOf": [
                {
                    "type": "object",
                    "properties": {
                        "action": {"const": "final"},
                        "answer": {"type": "string"},
                    },
                    "required": ["action", "answer"],
                    "additionalProperties": False,
                },
                {
                    "type": "object",
                    "properties": {
                        "action": {"const": "calculator"},
                        "input": {"type": "string"},
                    },
                    "required": ["action", "input"],
                    "additionalProperties": False,
                },
                {
                    "type": "object",
                    "properties": {
                        "action": {"const": "edit_file"},
                        "input": {"type": "object", "required": ["path"]},
                    },
                    "required": ["action", "input"],
                    "additionalProperties": False,
                },
            ]
        }

    def test_defaults_tool_input_to_unconstrained_when_absent(self) -> None:
        # Act
        schema = build_response_schema([tool_spec("echo")])

        # Assert: old tools remain usable until they declare a structured shape.
        assert schema.value == {
            "oneOf": [
                {
                    "type": "object",
                    "properties": {
                        "action": {"const": "final"},
                        "answer": {"type": "string"},
                    },
                    "required": ["action", "answer"],
                    "additionalProperties": False,
                },
                {
                    "type": "object",
                    "properties": {"action": {"const": "echo"}, "input": {}},
                    "required": ["action", "input"],
                    "additionalProperties": False,
                },
            ]
        }

    def test_without_tools_keeps_only_the_final_branch(self) -> None:
        # Act: an empty enum would be an unsatisfiable grammar, so drop the branch.
        schema = build_response_schema([])

        # Assert
        assert schema.value == {
            "type": "object",
            "properties": {
                "action": {"const": "final"},
                "answer": {"type": "string"},
            },
            "required": ["action", "answer"],
            "additionalProperties": False,
        }


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
            'Return only one valid JSON object with action="final".'
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
            "To call a tool:\n"
            '{"action": "<tool name>", "input": <tool input>}\n\n'
            "To give your final answer:\n"
            '{"action": "final", "answer": "..."}\n\n'
            "Do not return plain text. Do not return multiple JSON objects."
        )
        assert "144 divided by 12" not in message
