"""Request and result types for a single tool execution."""

from __future__ import annotations

from dataclasses import dataclass

from local_llm.domain.values.text_values import ToolInput, ToolName, ToolOutput


@dataclass(frozen=True)
class ToolRunRequest:
    """A request to run a named tool with raw model-provided input.

    Example:
        request = ToolRunRequest(ToolName("calculator"), ToolInput("2 + 2"))
    """

    tool_name: ToolName
    raw_input: ToolInput


@dataclass(frozen=True)
class ToolRunResult:
    """The outcome of running a tool: its output and whether it succeeded.

    Example:
        result = ToolRunResult(ToolName("calculator"), ToolOutput("4"), True)
    """

    tool_name: ToolName
    output: ToolOutput
    succeeded: bool
