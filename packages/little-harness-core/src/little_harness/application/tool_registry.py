"""First-class collection of the agent's tools, keyed by name."""

from __future__ import annotations

from collections.abc import Sequence

from little_harness.application.ports.agent_tool import AgentTool
from little_harness.domain.errors import ToolRegistrationError
from little_harness.domain.tool_spec import ToolSpec
from little_harness.domain.values.text_values import ToolName


class ToolRegistry:
    """Owns the name->tool mapping and rejects duplicate names at construction.

    Empty names cannot occur here: `ToolName` rejects them at its own boundary.

    Example:
        registry = ToolRegistry([CalculatorTool()])
        tool = registry.find(ToolName("calculator"))

    """

    def __init__(self, tools: Sequence[AgentTool]) -> None:
        """See class docstring for argument descriptions."""
        self._tools_by_name = build_tool_index(tools)

    def find(self, name: ToolName) -> AgentTool | None:
        """Look up a tool by name, or None if not registered."""
        return self._tools_by_name.get(name)

    def specs(self) -> tuple[ToolSpec, ...]:
        """Return the spec of every registered tool."""
        return tuple(tool.spec for tool in self._tools_by_name.values())

    def __len__(self) -> int:
        """Return the number of registered tools."""
        return len(self._tools_by_name)


def build_tool_index(tools: Sequence[AgentTool]) -> dict[ToolName, AgentTool]:
    """Build a name-to-tool dict, raising on duplicate names."""
    index: dict[ToolName, AgentTool] = {}

    for tool in tools:
        name = tool.spec.name

        if name in index:
            raise ToolRegistrationError(
                f"Duplicate tool name: {name.value}. Expected unique names."
            )

        index[name] = tool

    return index
