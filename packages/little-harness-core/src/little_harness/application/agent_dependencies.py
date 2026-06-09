"""Parameter object bundling the agent runtime's injected collaborators."""

from __future__ import annotations

from dataclasses import dataclass

from little_harness.application.ports.agent_observer import AgentObserver
from little_harness.application.ports.agent_policy import AgentPolicy
from little_harness.application.ports.chat_model import ChatModel
from little_harness.application.ports.lifecycle_hook import LifecycleHook
from little_harness.application.ports.token_sink import TokenSink
from little_harness.application.tool_registry import ToolRegistry


@dataclass(frozen=True)
class AgentDependencies:
    """The six ports the runtime depends on, grouped so it holds ≤2 fields.

    Example:
        dependencies = AgentDependencies(
            model, registry, policy, observer, sink, hooks
        )

    """

    chat_model: ChatModel
    tool_registry: ToolRegistry
    policy: AgentPolicy
    observer: AgentObserver
    token_sink: TokenSink
    hooks: LifecycleHook
