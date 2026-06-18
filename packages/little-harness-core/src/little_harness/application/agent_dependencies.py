"""Parameter object bundling the agent runtime's injected collaborators."""

from __future__ import annotations

from dataclasses import dataclass

from little_harness.application.ports.agent_observer import AgentObserver
from little_harness.application.ports.agent_policy import AgentPolicy
from little_harness.application.ports.chat_model import ChatModel
from little_harness.application.ports.lifecycle_hook import LifecycleHook
from little_harness.application.ports.skill_loader import SkillLoader
from little_harness.application.ports.token_sink import TokenSink
from little_harness.application.ports.tool_truncator import ToolTruncator
from little_harness.application.tool_registry import ToolRegistry
from little_harness.domain.values.truncation import TruncationConfig


@dataclass
class AgentDependencies:
    """The ports and config the runtime depends on.

    Example:
        dependencies = AgentDependencies(
            model, registry, policy, observer, sink, hooks, truncator, config
        )

    """

    chat_model: ChatModel
    tool_registry: ToolRegistry
    policy: AgentPolicy
    observer: AgentObserver
    token_sink: TokenSink
    hooks: LifecycleHook
    truncator: ToolTruncator
    truncation_config: TruncationConfig
    skill_loader: SkillLoader
