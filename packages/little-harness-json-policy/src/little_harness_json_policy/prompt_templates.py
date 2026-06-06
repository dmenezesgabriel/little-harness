"""Builds the flat-JSON prompts, response schema, and observation/repair messages.

The protocol is deliberately flat: `action` is either a tool name or "final",
with no separate discriminator. Small local models otherwise collapse a nested
`{"action":"tool","tool_name":...}` into `{"action":"edit_file"}`, so meeting
them where they already land removes a whole class of repair loops. Every example
shown is derived from the available tools, so this generic policy never hard-codes
a specific tool.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from little_harness.application.ports.chat_model import ResponseSchema
from little_harness.domain.tool_result import ToolRunResult
from little_harness.domain.tool_spec import ToolSpec
from little_harness.domain.values.text_values import MessageContent, Prompt

FINAL_ACTION = "final"
PLACEHOLDER_TOOL_NAME = "tool_name"
PLACEHOLDER_TOOL_INPUT = "tool input"


def render_system_prompt(tools: Sequence[ToolSpec]) -> MessageContent:
    rendered_tools = "\n".join(render_tool_spec(tool) for tool in tools)
    return MessageContent(
        system_prompt_text(rendered_tools, build_tool_call_example(tools))
    )


def render_tool_spec(tool: ToolSpec) -> str:
    examples = tool.input_schema.examples
    examples_text = (
        "" if examples.is_empty() else f" Examples: {examples.joined(', ')}."
    )
    return (
        f"{tool.name.value}: {tool.description} "
        f"Input: {tool.input_schema.description}.{examples_text}"
    )


def build_tool_call_example(tools: Sequence[ToolSpec]) -> str:
    """Render a concrete `{"action": ..., "input": ...}` from the first tool.

    The input is shown as valid JSON: a tool whose example already is a JSON
    object is inlined verbatim, while a bare expression is quoted as a string.
    """
    if len(tools) == 0:
        return tool_call_json(PLACEHOLDER_TOOL_NAME, json.dumps(PLACEHOLDER_TOOL_INPUT))

    first_tool = tools[0]
    examples = first_tool.input_schema.examples
    example = PLACEHOLDER_TOOL_INPUT if examples.is_empty() else examples.first()
    return tool_call_json(first_tool.name.value, format_input_example(example))


def format_input_example(example: str) -> str:
    try:
        json.loads(example)
    except json.JSONDecodeError:
        return json.dumps(example)

    return example


def tool_call_json(tool_name: str, input_json: str) -> str:
    return f'{{"action": "{tool_name}", "input": {input_json}}}'


def build_response_schema(tools: Sequence[ToolSpec]) -> ResponseSchema:
    """A JSON Schema forcing one branch: a final answer, or a known tool call.

    Each tool gets its own branch so constrained decoders can enforce both the
    tool name and the shape of that tool's input. With no tools, only the final
    branch remains.
    """
    if len(tools) == 0:
        return ResponseSchema(final_branch())

    return ResponseSchema({"oneOf": [final_branch(), *tool_branches(tools)]})


def final_branch() -> Mapping[str, object]:
    return {
        "type": "object",
        "properties": {"action": {"const": FINAL_ACTION}, "answer": {"type": "string"}},
        "required": ["action", "answer"],
        "additionalProperties": False,
    }


def tool_branches(tools: Sequence[ToolSpec]) -> list[Mapping[str, object]]:
    return [tool_branch(tool) for tool in tools]


def tool_branch(tool: ToolSpec) -> Mapping[str, object]:
    return {
        "type": "object",
        "properties": {
            "action": {"const": tool.name.value},
            "input": tool.input_schema.json_schema or {},
        },
        "required": ["action", "input"],
        "additionalProperties": False,
    }


def render_tool_observation(
    original_prompt: Prompt,
    tool_result: ToolRunResult,
) -> MessageContent:
    status = "succeeded" if tool_result.succeeded else "failed"
    return MessageContent(
        f"Original user question:\n{original_prompt.value}\n\n"
        f"Tool observation ({tool_result.tool_name.value}, {status}):\n"
        f"{tool_result.output.value}\n\n"
        "Now answer the full original user question.\n"
        'Return only one valid JSON object with action="final".'
    )


def render_repair_request(original_prompt: Prompt, error: Exception) -> MessageContent:
    return MessageContent(
        f"Your previous response was invalid. Error: {error}\n\n"
        f"Original user question:\n{original_prompt.value}\n\n"
        "Return only one valid JSON object.\n\n"
        "To call a tool:\n"
        '{"action": "<tool name>", "input": <tool input>}\n\n'
        "To give your final answer:\n"
        '{"action": "final", "answer": "..."}\n\n'
        "Do not return plain text. Do not return multiple JSON objects."
    )


def system_prompt_text(rendered_tools: str, tool_call_example: str) -> str:
    return f"""
You are an agent that answers a user's question through a loop of tool calls.

Available tools:

{rendered_tools}

Respond with exactly one JSON object and nothing else.

To call a tool:

{tool_call_example}

To give your final answer:

{{"action": "final", "answer": "<your complete answer>"}}

Rules:
- Return one JSON object only. No Markdown, no triple backticks, no plain text.
- "action" is the name of one available tool, or "final" to finish.
- A tool call needs "input"; a final answer needs "answer".
- Call a tool when it helps answer the question.
- After a tool observation, call another tool or give the final answer.
- The final answer must address every part of the original user question.
""".strip()
