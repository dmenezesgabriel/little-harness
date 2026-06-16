"""Entry-point builder for the find tool.

Registered under the ``little_harness.tools`` group as ``find``.
"""

from __future__ import annotations

from little_harness.application.ports.agent_tool import AgentTool

from little_harness_find.find_tool import FindTool


def build() -> AgentTool:
    """Build and return the find ``AgentTool``."""
    return FindTool()
