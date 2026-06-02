from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ToolInputSchema:
    description: str
    examples: tuple[str, ...] = ()


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: ToolInputSchema


@dataclass(frozen=True)
class ToolRunRequest:
    tool_name: str
    raw_input: str


@dataclass(frozen=True)
class ToolRunResult:
    tool_name: str
    output: str
    succeeded: bool


class AgentTool(Protocol):
    @property
    def spec(self) -> ToolSpec:
        """Describe the tool exposed to the agent.

        Example:
            tool_name = tool.spec.name
        """
        ...

    def run(self, request: ToolRunRequest) -> ToolRunResult:
        """Execute the tool for a raw model-provided input.

        Example:
            result = tool.run(ToolRunRequest("calculator", "2 + 2"))
        """
        ...
