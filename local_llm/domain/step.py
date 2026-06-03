"""A single recorded iteration of the agent loop."""

from __future__ import annotations

from dataclasses import dataclass

from local_llm.domain.decision import AgentDecision
from local_llm.domain.values.numeric_values import Iteration
from local_llm.domain.values.text_values import MessageContent


@dataclass(frozen=True)
class AgentStep:
    """What happened in one loop iteration.

    `decision` is None when the model output failed to parse and was repaired.

    Example:
        step = AgentStep(Iteration(1), output, decision, observation)
    """

    iteration: Iteration
    model_output: MessageContent
    decision: AgentDecision | None
    observation: MessageContent
