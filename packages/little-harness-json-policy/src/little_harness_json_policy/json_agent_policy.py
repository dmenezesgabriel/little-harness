"""The strict-JSON `AgentPolicy` adapter, composing parser and templates."""

from __future__ import annotations

from collections.abc import Sequence

from little_harness.domain.decision import AgentDecision
from little_harness.domain.message import ChatMessage
from little_harness.domain.tool_result import ToolRunResult
from little_harness.domain.tool_spec import ToolSpec
from little_harness.domain.values.role import USER
from little_harness.domain.values.text_values import MessageContent, Prompt

from little_harness_json_policy.decision_parser import JsonDecisionParser
from little_harness_json_policy.prompt_templates import (
    render_repair_request,
    render_system_prompt,
    render_tool_observation,
)


class JsonAgentPolicy:
    """Drives the model with a strict-JSON protocol.

    Example:
        decision = JsonAgentPolicy().parse_model_output(model_output)
    """

    def __init__(self) -> None:
        self._parser = JsonDecisionParser()

    def system_prompt(self, tools: Sequence[ToolSpec]) -> MessageContent:
        return render_system_prompt(tools)

    def parse_model_output(self, output: MessageContent) -> AgentDecision:
        return self._parser.parse(output)

    def build_tool_observation_message(
        self,
        original_prompt: Prompt,
        tool_result: ToolRunResult,
    ) -> ChatMessage:
        return ChatMessage(USER, render_tool_observation(original_prompt, tool_result))

    def build_repair_message(
        self,
        original_prompt: Prompt,
        error: Exception,
    ) -> ChatMessage:
        return ChatMessage(USER, render_repair_request(original_prompt, error))
