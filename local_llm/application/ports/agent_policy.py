"""Port for the strategy that drives the model's reasoning protocol."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from local_llm.domain.decision import AgentDecision
from local_llm.domain.message import ChatMessage
from local_llm.domain.tool_result import ToolRunResult
from local_llm.domain.tool_spec import ToolSpec
from local_llm.domain.values.text_values import MessageContent, Prompt


class AgentPolicy(Protocol):
    def system_prompt(self, tools: Sequence[ToolSpec]) -> MessageContent:
        """Return the system prompt for the available tools.

        Example:
            prompt = policy.system_prompt(registry.specs())
        """
        ...

    def parse_model_output(self, output: MessageContent) -> AgentDecision:
        """Parse model text into a validated agent decision.

        Example:
            decision = policy.parse_model_output(output)
        """
        ...

    def build_tool_observation_message(
        self,
        original_prompt: Prompt,
        tool_result: ToolRunResult,
    ) -> ChatMessage:
        """Return the next user message after a tool execution.

        Example:
            message = policy.build_tool_observation_message(prompt, result)
        """
        ...

    def build_repair_message(
        self,
        original_prompt: Prompt,
        error: Exception,
    ) -> ChatMessage:
        """Return a user message that asks the model to repair invalid output.

        Example:
            message = policy.build_repair_message(prompt, error)
        """
        ...
