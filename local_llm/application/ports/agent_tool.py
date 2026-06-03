"""Port for a tool the agent can call."""

from __future__ import annotations

from typing import Protocol

from local_llm.domain.tool_result import ToolRunRequest, ToolRunResult
from local_llm.domain.tool_spec import ToolSpec


class AgentTool(Protocol):
    @property
    def spec(self) -> ToolSpec:
        """Describe the tool exposed to the agent.

        Example:
            name = tool.spec.name
        """
        ...

    def run(self, request: ToolRunRequest) -> ToolRunResult:
        """Execute the tool for a raw model-provided input.

        Example:
            result = tool.run(request)
        """
        ...
