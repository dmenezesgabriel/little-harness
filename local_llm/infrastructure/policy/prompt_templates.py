"""Builds the strict-JSON prompts and observation/repair messages.

The example tool-call shown in the system prompt is derived from the available
tools, so this generic policy never hard-codes a specific tool. When no tool is
available it falls back to neutral placeholders.
"""

from __future__ import annotations

from collections.abc import Sequence

from local_llm.domain.tool_result import ToolRunResult
from local_llm.domain.tool_spec import ToolSpec
from local_llm.domain.values.text_values import MessageContent, Prompt

PLACEHOLDER_TOOL_NAME = "tool_name"
PLACEHOLDER_TOOL_INPUT = "tool input"


def render_system_prompt(tools: Sequence[ToolSpec]) -> MessageContent:
    rendered_tools = "\n".join(render_tool_spec(tool) for tool in tools)
    example_name, example_input = build_example_tool_call(tools)
    return MessageContent(
        system_prompt_text(rendered_tools, example_name, example_input)
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


def build_example_tool_call(tools: Sequence[ToolSpec]) -> tuple[str, str]:
    if len(tools) == 0:
        return PLACEHOLDER_TOOL_NAME, PLACEHOLDER_TOOL_INPUT

    first_tool = tools[0]
    examples = first_tool.input_schema.examples
    example_input = PLACEHOLDER_TOOL_INPUT if examples.is_empty() else examples.first()
    return first_tool.name.value, example_input


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
        "Return only one valid JSON object with action='final'."
    )


def render_repair_request(original_prompt: Prompt, error: Exception) -> MessageContent:
    return MessageContent(
        f"Your previous response was invalid. Error: {error}\n\n"
        f"Original user question:\n{original_prompt.value}\n\n"
        "Return only one valid JSON object.\n\n"
        "Use this schema for a final answer:\n"
        '{"action":"final","tool_name":null,"tool_input":null,"answer":"..."}'
        "\n\n"
        "Use this schema for a tool call:\n"
        '{"action":"tool","tool_name":"tool_name","tool_input":"tool input",'
        '"answer":null}\n\n'
        "Do not return plain text. Do not return multiple JSON objects."
    )


def system_prompt_text(
    rendered_tools: str,
    example_name: str,
    example_input: str,
) -> str:
    return f"""
You are a strict JSON agent running inside a local agent loop.

Available tools:

{rendered_tools}

You must always return exactly one valid JSON object.

Valid tool call:

{{
  "action": "tool",
  "tool_name": "{example_name}",
  "tool_input": "{example_input}",
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
- When an available tool can answer part of the question, call it first.
- After receiving a tool observation, answer the full original user question.
- The final answer must address every part of the original user question.
""".strip()
