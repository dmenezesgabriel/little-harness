"""Port for observing the agent loop — the seam for logging and tracing."""

from __future__ import annotations

from typing import Protocol

from local_llm.domain.decision import AgentDecision
from local_llm.domain.result import AgentResult
from local_llm.domain.tool_result import ToolRunResult
from local_llm.domain.values.numeric_values import Iteration
from local_llm.domain.values.text_values import MessageContent, Prompt


class AgentObserver(Protocol):
    """Receives lifecycle events. Implementations log, trace, or count them.

    The runtime emits events and never logs directly, so observability is added
    by swapping the observer rather than editing the loop.
    """

    def on_run_started(self, prompt: Prompt) -> None: ...

    def on_model_completed(
        self, iteration: Iteration, output: MessageContent
    ) -> None: ...

    def on_decision_parsed(
        self, iteration: Iteration, decision: AgentDecision
    ) -> None: ...

    def on_tool_invoked(self, iteration: Iteration, result: ToolRunResult) -> None: ...

    def on_repair(self, iteration: Iteration, error: Exception) -> None: ...

    def on_run_finished(self, result: AgentResult) -> None: ...
