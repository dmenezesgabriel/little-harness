"""The final outcome of an agent run."""

from __future__ import annotations

from dataclasses import dataclass

from local_llm.domain.steps import AgentSteps
from local_llm.domain.values.numeric_values import ElapsedSeconds
from local_llm.domain.values.text_values import MessageContent


@dataclass(frozen=True)
class AgentResult:
    """The agent's answer plus how long it took and the steps it took.

    Example:
        result = AgentResult(answer, ElapsedSeconds(1.2), steps)
    """

    answer: MessageContent
    elapsed: ElapsedSeconds
    steps: AgentSteps
