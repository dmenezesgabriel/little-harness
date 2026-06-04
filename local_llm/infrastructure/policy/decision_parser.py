"""Parses model JSON output into a typed `AgentDecision`.

This is the boundary where untyped text becomes a typed decision, so it inspects
the `action` discriminator directly before constructing the matching type.
"""

from __future__ import annotations

import json
from collections.abc import Mapping

from local_llm.domain.decision import AgentDecision, FinalAnswer, ToolCall
from local_llm.domain.errors import AgentProtocolError
from local_llm.domain.values.text_values import MessageContent, ToolInput, ToolName


class JsonDecisionParser:
    """Turns one JSON object embedded in model output into an `AgentDecision`.

    Example:
        decision = JsonDecisionParser().parse(MessageContent('{"action":...}'))
    """

    def parse(self, output: MessageContent) -> AgentDecision:
        return build_decision(extract_first_json_object(output.value))


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


def extract_first_json_object(text: str) -> Mapping[str, object]:
    """Decode the first brace-delimited JSON object, ignoring surrounding text.

    Anchoring on the first "{" both skips any prose the model emits and means a
    successful decode is always an object, so no separate object-type check is
    needed. `build_decision` validates the object's fields.
    """
    stripped = text.strip()
    start = stripped.find("{")

    if start == -1:
        raise AgentProtocolError(
            f"Could not find JSON object in model output: {text}. "
            "Expected one valid JSON object."
        )

    try:
        decoded, _ = json.JSONDecoder().raw_decode(stripped[start:])
    except json.JSONDecodeError as error:
        raise AgentProtocolError(
            f"Invalid JSON object in model output: {text}. "
            "Expected one valid JSON object."
        ) from error

    return decoded
