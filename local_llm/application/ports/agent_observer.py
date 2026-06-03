"""Port for observing the agent loop — the seam for logging and tracing."""

from __future__ import annotations

from typing import Protocol

from local_llm.domain.decision import AgentDecision
from local_llm.domain.result import AgentResult
from local_llm.domain.tool_result import ToolRunResult
from local_llm.domain.values.numeric_values import ElapsedSeconds, Iteration
from local_llm.domain.values.text_values import MessageContent, Prompt, RunId


class AgentObserver(Protocol):
    """Receives lifecycle events. Implementations log, trace, or count them.

    The runtime emits events and never logs directly, so observability is added
    by swapping the observer rather than editing the loop. Every event carries
    the run's `run_id` (the trace/span correlation key); model and tool events
    also carry the call's `elapsed` (the latency measurement).
    """

    def on_run_started(self, run_id: RunId, prompt: Prompt) -> None: ...

    def on_model_completed(
        self,
        run_id: RunId,
        iteration: Iteration,
        output: MessageContent,
        elapsed: ElapsedSeconds,
    ) -> None: ...

    def on_decision_parsed(
        self, run_id: RunId, iteration: Iteration, decision: AgentDecision
    ) -> None: ...

    def on_tool_invoked(
        self,
        run_id: RunId,
        iteration: Iteration,
        result: ToolRunResult,
        elapsed: ElapsedSeconds,
    ) -> None: ...

    def on_repair(
        self, run_id: RunId, iteration: Iteration, error: Exception
    ) -> None: ...

    def on_run_finished(self, run_id: RunId, result: AgentResult) -> None: ...
