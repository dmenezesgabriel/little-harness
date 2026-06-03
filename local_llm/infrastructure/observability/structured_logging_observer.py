"""AgentObserver that emits a structured log record for each lifecycle event."""

from __future__ import annotations

from local_llm.application.ports.structured_logger import StructuredLogger
from local_llm.domain.decision import AgentDecision
from local_llm.domain.result import AgentResult
from local_llm.domain.tool_result import ToolRunResult
from local_llm.domain.values.numeric_values import Iteration
from local_llm.domain.values.text_values import MessageContent, Prompt


class StructuredLoggingObserver:
    """Translates agent events into structured logs via a `StructuredLogger`.

    Example:
        observer = StructuredLoggingObserver(create_structured_logger("agent"))
    """

    def __init__(self, logger: StructuredLogger) -> None:
        self._logger = logger

    def on_run_started(self, prompt: Prompt) -> None:
        self._logger.log("run_started", {"prompt": prompt.value})

    def on_model_completed(self, iteration: Iteration, output: MessageContent) -> None:
        self._logger.log(
            "model_completed",
            {"iteration": iteration.value, "output_chars": len(output.value)},
        )

    def on_decision_parsed(self, iteration: Iteration, decision: AgentDecision) -> None:
        self._logger.log(
            "decision_parsed",
            {"iteration": iteration.value, "decision": type(decision).__name__},
        )

    def on_tool_invoked(self, iteration: Iteration, result: ToolRunResult) -> None:
        self._logger.log(
            "tool_invoked",
            {
                "iteration": iteration.value,
                "tool": result.tool_name.value,
                "succeeded": result.succeeded,
            },
        )

    def on_repair(self, iteration: Iteration, error: Exception) -> None:
        self._logger.log(
            "repair",
            {"iteration": iteration.value, "error": str(error)},
        )

    def on_run_finished(self, result: AgentResult) -> None:
        self._logger.log(
            "run_finished",
            {"elapsed_seconds": result.elapsed.value, "steps": len(result.steps)},
        )
