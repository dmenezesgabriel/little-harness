"""Mutable working state threaded through one agent run."""

from __future__ import annotations

from little_harness.domain.message import ChatMessage
from little_harness.domain.message_history import MessageHistory
from little_harness.domain.step import AgentStep
from little_harness.domain.steps import AgentSteps


class AgentLoopState:
    """Accumulates the conversation and the step trace as the loop advances.

    Example:
        state = AgentLoopState(initial_messages)
        state.append_message(observation)
    """

    def __init__(self, messages: MessageHistory) -> None:
        self._messages = messages
        self._steps = AgentSteps()

    def append_message(self, message: ChatMessage) -> None:
        self._messages = self._messages.with_message(message)

    def record_step(self, step: AgentStep) -> None:
        self._steps = self._steps.with_step(step)

    @property
    def messages(self) -> MessageHistory:
        return self._messages

    @property
    def steps(self) -> AgentSteps:
        return self._steps
