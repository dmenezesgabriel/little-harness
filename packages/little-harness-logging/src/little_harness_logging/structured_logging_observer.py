"""AgentObserver that emits a structured log record for each lifecycle event."""

from __future__ import annotations

from little_harness.domain.decision import AgentDecision
from little_harness.domain.result import AgentResult
from little_harness.domain.tool_result import ToolRunResult
from little_harness.domain.values.numeric_values import ElapsedSeconds, Iteration
from little_harness.domain.values.text_values import MessageContent, Prompt, RunId

from little_harness_logging.structured_logger import StructuredLogger


class StructuredLoggingObserver:
    """Translates agent events into structured logs via a `StructuredLogger`.

    Every record carries `run_id` so a run's events can be correlated.

    Example:
        observer = StructuredLoggingObserver(create_structured_logger("agent"))

    """

    def __init__(self, logger: StructuredLogger) -> None:
        """See class docstring for argument descriptions."""
        self._logger = logger

    def on_run_started(self, run_id: RunId, prompt: Prompt) -> None:
        """Log a run_started event with the prompt text."""
        self._logger.log(
            "run_started", {"run_id": run_id.value, "prompt": prompt.value}
        )

    def on_model_completed(
        self,
        run_id: RunId,
        iteration: Iteration,
        output: MessageContent,
        elapsed: ElapsedSeconds,
    ) -> None:
        """Log a model_completed event with output length and timing."""
        self._logger.log(
            "model_completed",
            {
                "run_id": run_id.value,
                "iteration": iteration.value,
                "output_chars": len(output.value),
                "elapsed_seconds": elapsed.value,
            },
        )

    def on_decision_parsed(
        self, run_id: RunId, iteration: Iteration, decision: AgentDecision
    ) -> None:
        """Log a decision_parsed event with the decision type."""
        self._logger.log(
            "decision_parsed",
            {
                "run_id": run_id.value,
                "iteration": iteration.value,
                "decision": type(decision).__name__,
            },
        )

    def on_tool_invoked(
        self,
        run_id: RunId,
        iteration: Iteration,
        result: ToolRunResult,
        elapsed: ElapsedSeconds,
    ) -> None:
        """Log a tool_invoked event with the tool name and success status."""
        self._logger.log(
            "tool_invoked",
            {
                "run_id": run_id.value,
                "iteration": iteration.value,
                "tool": result.tool_name.value,
                "succeeded": result.succeeded,
                "elapsed_seconds": elapsed.value,
            },
        )

    def on_repair(self, run_id: RunId, iteration: Iteration, error: Exception) -> None:
        """Log a repair event with the error message."""
        self._logger.log(
            "repair",
            {"run_id": run_id.value, "iteration": iteration.value, "error": str(error)},
        )

    def on_run_finished(self, run_id: RunId, result: AgentResult) -> None:
        """Log a run_finished event with elapsed time and step count."""
        self._logger.log(
            "run_finished",
            {
                "run_id": run_id.value,
                "elapsed_seconds": result.elapsed.value,
                "steps": len(result.steps),
            },
        )
