from __future__ import annotations

import json
from collections.abc import Sequence
from typing import cast

from local_llm.agent import AgentActionKind, AgentDecision, AgentProtocolError
from local_llm.chat import ChatMessage
from local_llm.tools import ToolRunResult, ToolSpec


class JsonAgentPolicy:
    def system_prompt(self, tools: Sequence[ToolSpec]) -> str:
        rendered_tools = "\n".join(render_tool_spec(tool) for tool in tools)
        return create_system_prompt(rendered_tools)

    def parse_model_output(self, output: str) -> AgentDecision:
        json_text = extract_first_json_object(output)

        try:
            parsed = json.loads(json_text)
        except json.JSONDecodeError as error:
            raise AgentProtocolError(
                f"Invalid JSON object in model output: {output}. "
                "Expected one valid JSON object."
            ) from error

        if not isinstance(parsed, dict):
            raise AgentProtocolError(f"Expected JSON object, got: {type(parsed)}")

        return parse_decision_object(cast("dict[object, object]", parsed))

    def build_tool_observation_message(
        self,
        original_prompt: str,
        tool_result: ToolRunResult,
    ) -> ChatMessage:
        status = "succeeded" if tool_result.succeeded else "failed"

        return ChatMessage(
            role="user",
            content=create_tool_observation(original_prompt, tool_result, status),
        )

    def build_repair_message(
        self,
        original_prompt: str,
        error: Exception,
    ) -> ChatMessage:
        return ChatMessage(
            role="user",
            content=create_repair_observation(original_prompt, error),
        )


def create_system_prompt(rendered_tools: str) -> str:
    return f"""
You are a strict JSON agent running inside a local agent loop.

Available tools:

{rendered_tools}

You must always return exactly one valid JSON object.

Valid tool call:

{{
  "action": "tool",
  "tool_name": "calculator",
  "tool_input": "144 / 12",
  "answer": null
}}

Valid final answer:

{{
  "action": "final",
  "tool_name": null,
  "tool_input": null,
  "answer": "144 divided by 12 is 12. 12 is even."
}}

Rules:
- Return JSON only.
- Do not return Markdown.
- Do not use triple backticks.
- Do not return plain text.
- Do not return more than one JSON object.
- For arithmetic, call the calculator first.
- After receiving a tool observation, answer the full original user question.
- The final answer must address every part of the original user question.
""".strip()


def create_tool_observation(
    original_prompt: str,
    tool_result: ToolRunResult,
    status: str,
) -> str:
    return (
        f"Original user question:\n{original_prompt}\n\n"
        f"Tool observation ({tool_result.tool_name}, {status}):\n"
        f"{tool_result.output}\n\n"
        "Now answer the full original user question.\n"
        "Return only one valid JSON object with action='final'."
    )


def create_repair_observation(original_prompt: str, error: Exception) -> str:
    return (
        f"Your previous response was invalid. Error: {error}\n\n"
        f"Original user question:\n{original_prompt}\n\n"
        "Return only one valid JSON object.\n\n"
        "Use this schema for a final answer:\n"
        '{"action":"final","tool_name":null,"tool_input":null,"answer":"..."}'
        "\n\n"
        "Use this schema for a tool call:\n"
        '{"action":"tool","tool_name":"calculator","tool_input":"2 + 2",'
        '"answer":null}\n\n'
        "Do not return plain text. Do not return multiple JSON objects."
    )


def render_tool_spec(tool: ToolSpec) -> str:
    examples = ", ".join(tool.input_schema.examples)
    examples_text = f" Examples: {examples}." if examples else ""

    return (
        f"{tool.name}: {tool.description} "
        f"Input: {tool.input_schema.description}.{examples_text}"
    )


def parse_decision_object(parsed: dict[object, object]) -> AgentDecision:
    action = parsed.get("action")
    tool_name = parsed.get("tool_name")
    tool_input = parsed.get("tool_input")
    answer = parsed.get("answer")

    validate_decision_shape(action, tool_name, tool_input, answer)

    return AgentDecision(
        kind=cast("AgentActionKind", action),
        tool_name=cast("str | None", tool_name),
        tool_input=cast("str | None", tool_input),
        final_answer=answer.strip() if isinstance(answer, str) else None,
    )


def validate_decision_shape(
    action: object,
    tool_name: object,
    tool_input: object,
    answer: object,
) -> None:
    if action not in ("tool", "final"):
        raise AgentProtocolError(f"Expected action 'tool' or 'final', got: {action}")

    validate_optional_string_field("tool_name", tool_name)
    validate_optional_string_field("tool_input", tool_input)
    validate_optional_string_field("answer", answer)
    require_action_payload(action, tool_name, tool_input, answer)


def validate_optional_string_field(field_name: str, value: object) -> None:
    if value is None or isinstance(value, str):
        return

    raise AgentProtocolError(
        f"Expected {field_name} string or null, got: {type(value)}"
    )


def require_action_payload(
    action: object,
    tool_name: object,
    tool_input: object,
    answer: object,
) -> None:
    if action == "final":
        require_final_answer(answer)
        return

    require_tool_call_payload(tool_name, tool_input)


def require_final_answer(answer: object) -> None:
    if answer is not None:
        return

    raise AgentProtocolError(
        f"Final action answer is invalid: {answer}. Expected non-null string."
    )


def require_tool_call_payload(tool_name: object, tool_input: object) -> None:
    require_tool_name_payload(tool_name)
    require_tool_input_payload(tool_input)


def require_tool_name_payload(tool_name: object) -> None:
    if tool_name is not None:
        return

    raise AgentProtocolError(
        f"Tool action tool_name is invalid: {tool_name}. Expected non-null string."
    )


def require_tool_input_payload(tool_input: object) -> None:
    if tool_input is not None:
        return

    raise AgentProtocolError(
        f"Tool action tool_input is invalid: {tool_input}. Expected non-null string."
    )


def extract_first_json_object(text: str) -> str:
    stripped = text.strip()
    start = stripped.find("{")

    if start == -1:
        raise AgentProtocolError(
            f"Could not find JSON object in model output: {text}. "
            "Expected one valid JSON object."
        )

    decoder = json.JSONDecoder()

    try:
        _, end = decoder.raw_decode(stripped[start:])
    except json.JSONDecodeError as error:
        raise AgentProtocolError(
            f"Invalid JSON object in model output: {text}. "
            "Expected one valid JSON object."
        ) from error

    return stripped[start : start + end]
