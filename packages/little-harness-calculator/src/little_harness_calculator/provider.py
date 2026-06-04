"""Entry-point builder for the calculator tool.

Registered under the `little_harness.tools` group as `calculator`. The core
composition root calls `build()` once per discovered tool and registers the
result in the `ToolRegistry`.

Example:
    tool = build()
"""

from __future__ import annotations

from little_harness.application.ports.agent_tool import AgentTool

from little_harness_calculator.calculator_tool import CalculatorTool


def build() -> AgentTool:
    return CalculatorTool()
