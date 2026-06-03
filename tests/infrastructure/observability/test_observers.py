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


class TestNullObserver:
    def test_conforms_to_the_observer_port_and_does_nothing(self) -> None:
        # Arrange: the explicit annotation forces a protocol-conformance check.
        observer: AgentObserver = NullObserver()

        # Act / Assert: every call is a no-op that returns None.
        assert observer.on_run_started(Prompt("q")) is None
        assert observer.on_repair(Iteration(1), ValueError("x")) is None


class TestStructuredLoggingObserver:
    def test_logs_run_started_with_the_prompt(self) -> None:
        # Arrange
        logger = RecordingLogger()
        observer = StructuredLoggingObserver(logger)

        # Act
        observer.on_run_started(Prompt("what is 2 + 2?"))

        # Assert
        assert logger.records == [("run_started", {"prompt": "what is 2 + 2?"})]

    def test_logs_tool_invocation_with_outcome(self) -> None:
        # Arrange
        logger = RecordingLogger()
        observer = StructuredLoggingObserver(logger)
        result = ToolRunResult(ToolName("calculator"), ToolOutput("4"), succeeded=True)

        # Act
        observer.on_tool_invoked(Iteration(2), result)

        # Assert
        assert logger.records == [
            (
                "tool_invoked",
                {"iteration": 2, "tool": "calculator", "succeeded": True},
            )
        ]

    def test_logs_model_decision_and_repair_events(self) -> None:
        # Arrange
        logger = RecordingLogger()
        observer = StructuredLoggingObserver(logger)

        # Act
        observer.on_model_completed(Iteration(1), MessageContent("hello"))
        observer.on_decision_parsed(Iteration(1), FinalAnswer(MessageContent("done")))
        observer.on_repair(Iteration(2), ValueError("bad output"))

        # Assert
        assert logger.records == [
            ("model_completed", {"iteration": 1, "output_chars": 5}),
            ("decision_parsed", {"iteration": 1, "decision": "FinalAnswer"}),
            ("repair", {"iteration": 2, "error": "bad output"}),
        ]

    def test_logs_run_finished_with_elapsed_and_step_count(self) -> None:
        # Arrange
        logger = RecordingLogger()
        observer = StructuredLoggingObserver(logger)
        result = AgentResult(MessageContent("done"), ElapsedSeconds(1.5), AgentSteps())

        # Act
        observer.on_run_finished(result)

        # Assert
        assert logger.records == [
            ("run_finished", {"elapsed_seconds": 1.5, "steps": 0})
        ]
