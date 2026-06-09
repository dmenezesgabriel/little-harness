"""Port for a tool the agent can call."""

from __future__ import annotations

from typing import Protocol

from little_harness.domain.tool_result import ToolRunRequest, ToolRunResult
from little_harness.domain.tool_spec import ToolSpec


class AgentTool(Protocol):
    """Describe a tool the agent can call."""

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
