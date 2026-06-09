"""Entry-point builder for the ripgrep tool.

Registered under the `little_harness.tools` group as `ripgrep`. The composition
root calls `build()` once and registers the result in the `ToolRegistry`.

Example:
    tool = build()

"""

from __future__ import annotations

from little_harness.application.ports.agent_tool import AgentTool

from little_harness_ripgrep.ripgrep_search import PythonGrepSearch
from little_harness_ripgrep.ripgrep_tool import RipgrepTool


def build() -> AgentTool:
    """Build and return a RipgrepTool instance with a PythonGrepSearch backend.

    Example:
        tool = build()

    """
    return RipgrepTool(PythonGrepSearch())
