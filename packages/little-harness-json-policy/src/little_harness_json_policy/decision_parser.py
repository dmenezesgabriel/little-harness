"""Parses model JSON output into a typed `AgentDecision`.

This is the boundary where untyped text becomes a typed decision. It is lenient
on purpose: small local models flatten the protocol, so any non-"final" action is
read as a tool name (whether the model wrote `{"action":"edit_file"}` directly or
the older `{"action":"tool","tool_name":"edit_file"}`), and a tool input may
arrive as a JSON string or as a nested object.
"""

from __future__ import annotations

import json
from collections.abc import Mapping

from little_harness.domain.decision import AgentDecision, FinalAnswer, ToolCall
from little_harness.domain.errors import AgentProtocolError
from little_harness.domain.values.text_values import MessageContent, ToolInput, ToolName

from little_harness_json_policy.prompt_templates import FINAL_ACTION

LEGACY_TOOL_ACTION = "tool"
# Keys that name the decision itself, never part of a tool's input. Everything
# else at the top level is treated as inline tool arguments (see resolve_tool_input).
DECISION_KEYS = frozenset({"action", "answer", "tool_name", "input", "tool_input"})


class JsonDecisionParser:
    """Turns one JSON object embedded in model output into an `AgentDecision`.

    Example:
        decision = JsonDecisionParser().parse(MessageContent('{"action":...}'))
    """

    def parse(self, output: MessageContent) -> AgentDecision:
        return build_decision(extract_first_json_object(output.value))


def build_decision(parsed: Mapping[str, object]) -> AgentDecision:
    action = parsed.get("action")

    if action == FINAL_ACTION:
        return FinalAnswer(
            MessageContent(require_string_field(parsed, "answer").strip())
        )

    if isinstance(action, str) and action.strip():
        return ToolCall(
            to_tool_name(resolve_tool_name(parsed, action)),
            resolve_tool_input(parsed),
        )

    raise AgentProtocolError(
        f"Expected a tool name or {FINAL_ACTION!r} as action, got: {action!r}."
    )


def resolve_tool_name(parsed: Mapping[str, object], action: str) -> str:
    # `action == "tool"` is the older nested protocol; the name lives elsewhere.
    if action == LEGACY_TOOL_ACTION:
        return require_string_field(parsed, "tool_name")

    return action


def resolve_tool_input(parsed: Mapping[str, object]) -> ToolInput:
    raw = parsed.get("input", parsed.get("tool_input"))

    if isinstance(raw, str):
        return ToolInput(raw)

    if isinstance(raw, Mapping):
        return ToolInput(json.dumps(raw))

    if raw is None:
        return tool_input_from_top_level(parsed)

    raise AgentProtocolError(
        f"Tool input is invalid: {raw!r}. Expected a JSON string or object."
    )


def tool_input_from_top_level(parsed: Mapping[str, object]) -> ToolInput:
    """Treat the object's non-decision keys as inline tool arguments.

    Small models often skip the `input` wrapper and write the arguments beside
    `action`, e.g. {"action":"write_file","path":"a.txt","content":"hi"}. Folding
    those keys into the input keeps such replies usable.
    """
    arguments = {
        key: value for key, value in parsed.items() if key not in DECISION_KEYS
    }

    if arguments:
        return ToolInput(json.dumps(arguments))

    raise AgentProtocolError(
        "Tool input is invalid: no 'input' field and no inline arguments. "
        "Expected a JSON string, a JSON object, or arguments beside 'action'."
    )


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
