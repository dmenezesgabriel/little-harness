"""Entry-point builder for the strict-JSON agent policy.

Registered under the `little_harness.agent_policies` group as `json`. The core
composition root calls `build()` once and injects the result as the runtime's
`AgentPolicy`, so the reasoning protocol is selected without any core edit.

Example:
    policy = build()
"""

from __future__ import annotations

from little_harness.application.ports.agent_policy import AgentPolicy

from little_harness_json_policy.json_agent_policy import JsonAgentPolicy


def build() -> AgentPolicy:
    return JsonAgentPolicy()
