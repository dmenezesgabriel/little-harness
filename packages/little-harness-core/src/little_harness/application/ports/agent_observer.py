"""Port for observing the agent loop — the seam for logging and tracing."""

from __future__ import annotations

from typing import Protocol

from little_harness.domain.decision import AgentDecision
from little_harness.domain.result import AgentResult
from little_harness.domain.tool_result import ToolRunResult
from little_harness.domain.values.numeric_values import ElapsedSeconds, Iteration
from little_harness.domain.values.text_values import MessageContent, Prompt, RunId


class AgentObserver(Protocol):
    """Receives lifecycle events. Implementations log, trace, or count them.

    The runtime emits events and never logs directly, so observability is added
    by swapping the observer rather than editing the loop. Every event carries
    the run's `run_id` (the trace/span correlation key); model and tool events
    also carry the call's `elapsed` (the latency measurement).
    """

    def on_run_started(self, run_id: RunId, prompt: Prompt) -> None:
        """Record the start of a new agent run."""

    def on_model_completed(
        self,
        run_id: RunId,
        iteration: Iteration,
        output: MessageContent,
        elapsed: ElapsedSeconds,
    ) -> None:
        """Record the model completion output."""

    def on_decision_parsed(
        self, run_id: RunId, iteration: Iteration, decision: AgentDecision
    ) -> None:
        """Record a parsed agent decision."""

    def on_tool_invoked(
        self,
        run_id: RunId,
        iteration: Iteration,
        result: ToolRunResult,
        elapsed: ElapsedSeconds,
    ) -> None:
        """Record a tool invocation result."""

    def on_repair(self, run_id: RunId, iteration: Iteration, error: Exception) -> None:
        """Record a repair attempt for invalid model output."""

    def on_run_finished(self, run_id: RunId, result: AgentResult) -> None:
        """Record the finish of an agent run."""
