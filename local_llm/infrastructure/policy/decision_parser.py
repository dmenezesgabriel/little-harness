"""Parses model JSON output into a typed `AgentDecision`.

This is the boundary where untyped text becomes a typed decision, so it inspects
the `action` discriminator directly before constructing the matching type.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import cast

from local_llm.domain.decision import AgentDecision, FinalAnswer, ToolCall
from local_llm.domain.errors import AgentProtocolError
from local_llm.domain.values.text_values import MessageContent, ToolInput, ToolName


class JsonDecisionParser:
    """Turns one JSON object embedded in model output into an `AgentDecision`.

    Example:
        decision = JsonDecisionParser().parse(MessageContent('{"action":...}'))
    """

    def parse(self, output: MessageContent) -> AgentDecision:
        json_text = extract_first_json_object(output.value)
        parsed = load_json_object(json_text, output.value)
        return build_decision(parsed)


def build_decision(parsed: Mapping[str, object]) -> AgentDecision:
    action = parsed.get("action")

    if action == "final":
        return FinalAnswer(
            MessageContent(require_string_field(parsed, "answer").strip())
        )

    if action == "tool":
        return ToolCall(
            to_tool_name(require_string_field(parsed, "tool_name")),
            ToolInput(require_string_field(parsed, "tool_input")),
        )

    raise AgentProtocolError(f"Expected action 'tool' or 'final', got: {action}")


def require_string_field(parsed: Mapping[str, object], field_name: str) -> str:
    value = parsed.get(field_name)

    if isinstance(value, str):
        return value

    raise AgentProtocolError(
        f"Field {field_name} is invalid: {value!r}. Expected a non-null string."
    )


def to_tool_name(value: str) -> ToolName:
    try:
        return ToolName(value)
    except ValueError as error:
        raise AgentProtocolError(
            f"Invalid tool name in model output: {value!r}. "
            f"Expected a non-empty tool name ({error})."
        ) from error


def load_json_object(json_text: str, original: str) -> Mapping[str, object]:
    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError as error:
        raise AgentProtocolError(
            f"Invalid JSON object in model output: {original}. "
            "Expected one valid JSON object."
        ) from error

    if not isinstance(parsed, dict):
        raise AgentProtocolError(f"Expected JSON object, got: {type(parsed)}")

    return cast("Mapping[str, object]", parsed)


def extract_first_json_object(text: str) -> str:
    stripped = text.strip()
    start = stripped.find("{")

    if start == -1:
        raise AgentProtocolError(
            f"Could not find JSON object in model output: {text}. "
            "Expected one valid JSON object."
        )

    try:
        _, end = json.JSONDecoder().raw_decode(stripped[start:])
    except json.JSONDecodeError as error:
        raise AgentProtocolError(
            f"Invalid JSON object in model output: {text}. "
            "Expected one valid JSON object."
        ) from error

    return stripped[start : start + end]
