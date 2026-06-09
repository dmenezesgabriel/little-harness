"""No-op `AgentObserver`: the default when observability is not configured.

Also the extension point: subclass and override only the events you care
about, the way `NullHook` keeps `LifecycleHook` implementations selective.
Precise signatures (not variadic) let pyright verify subclass overrides.
"""

from __future__ import annotations

from little_harness.domain.decision import AgentDecision
from little_harness.domain.result import AgentResult
from little_harness.domain.tool_result import ToolRunResult
from little_harness.domain.values.numeric_values import ElapsedSeconds, Iteration
from little_harness.domain.values.text_values import MessageContent, Prompt, RunId


class NullObserver:
    """Discards every event so the runtime can always hold a real observer.

    The Null Object of `AgentObserver`. Subclass and override only the events
    you care about; every other point is silently discarded.

    Example:
        class MyObserver(NullObserver):
            def on_run_started(self, run_id: RunId, prompt: Prompt) -> None:
                print(f"Run {run_id.value} started")

    """

    # Parameter names mirror the AgentObserver Protocol exactly so pyright's
    # strict structural-subtype check accepts this class wherever the protocol
    # is expected. Leading underscores are intentionally absent.
    def on_run_started(self, run_id: RunId, prompt: Prompt) -> None:
        """Ignore run start event."""
        del run_id, prompt

    def on_model_completed(
        self,
        run_id: RunId,
        iteration: Iteration,
        output: MessageContent,
        elapsed: ElapsedSeconds,
    ) -> None:
        """Ignore model completion event."""
        del run_id, iteration, output, elapsed

    def on_decision_parsed(
        self, run_id: RunId, iteration: Iteration, decision: AgentDecision
    ) -> None:
        """Ignore decision parsed event."""
        del run_id, iteration, decision

    def on_tool_invoked(
        self,
        run_id: RunId,
        iteration: Iteration,
        result: ToolRunResult,
        elapsed: ElapsedSeconds,
    ) -> None:
        """Ignore tool invoked event."""
        del run_id, iteration, result, elapsed

    def on_repair(self, run_id: RunId, iteration: Iteration, error: Exception) -> None:
        """Ignore repair event."""
        del run_id, iteration, error

    def on_run_finished(self, run_id: RunId, result: AgentResult) -> None:
        """Ignore run finished event."""
        del run_id, result
