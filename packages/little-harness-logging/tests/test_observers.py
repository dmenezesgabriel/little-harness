from __future__ import annotations

from collections.abc import Mapping

from little_harness.domain.decision import FinalAnswer
from little_harness.domain.result import AgentResult
from little_harness.domain.steps import AgentSteps
from little_harness.domain.tool_result import ToolRunResult
from little_harness.domain.values.model_call_metrics import ModelCallMetrics
from little_harness.domain.values.numeric_values import ElapsedSeconds, Iteration
from little_harness.domain.values.text_values import (
    MessageContent,
    Prompt,
    RunId,
    ToolName,
    ToolOutput,
)
from little_harness_logging.structured_logging_observer import StructuredLoggingObserver


class RecordingLogger:
    """StructuredLogger fake that records each event and its fields."""

    def __init__(self) -> None:
        self.records: list[tuple[str, Mapping[str, object]]] = []

    def log(self, event: str, fields: Mapping[str, object]) -> None:
        self.records.append((event, fields))


RUN_ID = RunId("run-1")


class TestStructuredLoggingObserver:
    def test_logs_run_started_with_run_id_and_prompt(self) -> None:
        # Arrange
        logger = RecordingLogger()
        observer = StructuredLoggingObserver(logger)

        # Act
        observer.on_run_started(RUN_ID, Prompt("what is 2 + 2?"))

        # Assert
        assert logger.records == [
            ("run_started", {"run_id": "run-1", "prompt": "what is 2 + 2?"})
        ]

    def test_logs_tool_invocation_with_outcome_and_elapsed(self) -> None:
        # Arrange
        logger = RecordingLogger()
        observer = StructuredLoggingObserver(logger)
        result = ToolRunResult(ToolName("calculator"), ToolOutput("4"), succeeded=True)

        # Act
        observer.on_tool_invoked(RUN_ID, Iteration(2), result, ElapsedSeconds(0.25))

        # Assert
        assert logger.records == [
            (
                "tool_invoked",
                {
                    "run_id": "run-1",
                    "iteration": 2,
                    "tool": "calculator",
                    "succeeded": True,
                    "elapsed_seconds": 0.25,
                },
            )
        ]

    def test_logs_model_decision_and_repair_events(self) -> None:
        # Arrange
        logger = RecordingLogger()
        observer = StructuredLoggingObserver(logger)

        # Act
        observer.on_model_completed(
            RUN_ID, Iteration(1), MessageContent("hello"), ElapsedSeconds(0.5)
        )
        observer.on_decision_parsed(
            RUN_ID, Iteration(1), FinalAnswer(MessageContent("done"))
        )
        observer.on_repair(RUN_ID, Iteration(2), ValueError("bad output"))

        # Assert
        assert logger.records == [
            (
                "model_completed",
                {
                    "run_id": "run-1",
                    "iteration": 1,
                    "output_chars": 5,
                    "elapsed_seconds": 0.5,
                },
            ),
            (
                "decision_parsed",
                {"run_id": "run-1", "iteration": 1, "decision": "FinalAnswer"},
            ),
            ("repair", {"run_id": "run-1", "iteration": 2, "error": "bad output"}),
        ]

    def test_logs_model_metrics_with_ttft_tokens_and_throughput(self) -> None:
        # Arrange
        logger = RecordingLogger()
        observer = StructuredLoggingObserver(logger)
        metrics = ModelCallMetrics(
            elapsed=ElapsedSeconds(2.0),
            time_to_first_token=ElapsedSeconds(0.4),
            output_tokens=20,
        )

        # Act
        observer.on_model_metrics(RUN_ID, Iteration(1), metrics)

        # Assert
        assert logger.records == [
            (
                "model_metrics",
                {
                    "run_id": "run-1",
                    "iteration": 1,
                    "time_to_first_token_seconds": 0.4,
                    "output_tokens": 20,
                    "tokens_per_second": 10.0,
                    "elapsed_seconds": 2.0,
                },
            )
        ]

    def test_logs_model_metrics_with_null_ttft_when_absent(self) -> None:
        # Arrange
        logger = RecordingLogger()
        observer = StructuredLoggingObserver(logger)
        metrics = ModelCallMetrics(
            elapsed=ElapsedSeconds(1.0),
            time_to_first_token=None,
            output_tokens=0,
        )

        # Act
        observer.on_model_metrics(RUN_ID, Iteration(1), metrics)

        # Assert
        event, fields = logger.records[0]
        assert event == "model_metrics"
        assert fields["time_to_first_token_seconds"] is None

    def test_logs_run_finished_with_elapsed_and_step_count(self) -> None:
        # Arrange
        logger = RecordingLogger()
        observer = StructuredLoggingObserver(logger)
        result = AgentResult(MessageContent("done"), ElapsedSeconds(1.5), AgentSteps())

        # Act
        observer.on_run_finished(RUN_ID, result)

        # Assert
        assert logger.records == [
            ("run_finished", {"run_id": "run-1", "elapsed_seconds": 1.5, "steps": 0})
        ]
