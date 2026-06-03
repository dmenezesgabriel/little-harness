from __future__ import annotations

from collections.abc import Mapping

from local_llm.application.ports.agent_observer import AgentObserver
from local_llm.domain.decision import FinalAnswer
from local_llm.domain.result import AgentResult
from local_llm.domain.steps import AgentSteps
from local_llm.domain.tool_result import ToolRunResult
from local_llm.domain.values.numeric_values import ElapsedSeconds, Iteration
from local_llm.domain.values.text_values import (
    MessageContent,
    Prompt,
    RunId,
    ToolName,
    ToolOutput,
)
from local_llm.infrastructure.observability.null_observer import NullObserver
from local_llm.infrastructure.observability.structured_logging_observer import (
    StructuredLoggingObserver,
)


class RecordingLogger:
    """StructuredLogger fake that records each event and its fields."""

    def __init__(self) -> None:
        self.records: list[tuple[str, Mapping[str, object]]] = []

    def log(self, event: str, fields: Mapping[str, object]) -> None:
        self.records.append((event, fields))


RUN_ID = RunId("run-1")


class TestNullObserver:
    def test_conforms_to_the_observer_port_and_does_nothing(self) -> None:
        # Arrange: the explicit annotation forces a protocol-conformance check.
        observer: AgentObserver = NullObserver()

        # Act / Assert: every call is a no-op that returns None.
        assert observer.on_run_started(RUN_ID, Prompt("q")) is None
        assert observer.on_repair(RUN_ID, Iteration(1), ValueError("x")) is None


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
