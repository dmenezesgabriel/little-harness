# ruff: noqa: D100, D101, D102, D103, D107
from little_harness.domain.decision import AgentDecision
from little_harness.domain.result import AgentResult
from little_harness.domain.tool_result import ToolRunResult
from little_harness.domain.values.numeric_values import ElapsedSeconds, Iteration
from little_harness.domain.values.text_values import (
    MessageContent,
    Prompt,
    RunId,
    SessionId,
)

from little_harness_session_jsonl.infrastructure.jsonl_appender import JsonlFileAppender


class JsonlSessionObserver:
    """Observes agent lifecycle events and appends them to a JSONL file."""

    def __init__(self, session_id: SessionId, appender: JsonlFileAppender) -> None:
        self._session_id = session_id
        self._appender = appender

    def on_run_started(self, run_id: RunId, prompt: Prompt) -> None:
        self._appender.append(
            {
                "session_id": self._session_id.value,
                "type": "run_started",
                "run_id": run_id.value,
                "prompt": prompt.value,
            }
        )

    def on_model_completed(
        self,
        run_id: RunId,
        iteration: Iteration,
        output: MessageContent,
        elapsed: ElapsedSeconds,
    ) -> None:
        self._appender.append(
            {
                "session_id": self._session_id.value,
                "type": "model_completed",
                "run_id": run_id.value,
                "iteration": iteration.value,
                "output": output.value,
                "elapsed": elapsed.value,
            }
        )

    def on_decision_parsed(
        self, run_id: RunId, iteration: Iteration, decision: AgentDecision
    ) -> None:
        self._appender.append(
            {
                "session_id": self._session_id.value,
                "type": "decision_parsed",
                "run_id": run_id.value,
                "iteration": iteration.value,
                "decision": type(decision).__name__,
            }
        )

    def on_tool_invoked(
        self,
        run_id: RunId,
        iteration: Iteration,
        result: ToolRunResult,
        elapsed: ElapsedSeconds,
    ) -> None:
        self._appender.append(
            {
                "session_id": self._session_id.value,
                "type": "tool_invoked",
                "run_id": run_id.value,
                "iteration": iteration.value,
                "tool_name": result.tool_name.value,
                "output": result.output.value,
                "succeeded": result.succeeded,
                "elapsed": elapsed.value,
            }
        )

    def on_repair(self, run_id: RunId, iteration: Iteration, error: Exception) -> None:
        self._appender.append(
            {
                "session_id": self._session_id.value,
                "type": "repair",
                "run_id": run_id.value,
                "iteration": iteration.value,
                "error": type(error).__name__,
                "message": str(error),
            }
        )

    def on_run_finished(self, run_id: RunId, result: AgentResult) -> None:
        self._appender.append(
            {
                "session_id": self._session_id.value,
                "type": "run_finished",
                "run_id": run_id.value,
                "answer": result.answer.value,
                "elapsed": result.elapsed.value,
            }
        )
