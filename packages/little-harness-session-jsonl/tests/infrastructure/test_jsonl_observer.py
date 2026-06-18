# ruff: noqa: D100, D101, D102, D103
import json
from pathlib import Path

from little_harness.domain.decision import FinalAnswer
from little_harness.domain.errors import AgentProtocolError
from little_harness.domain.result import AgentResult
from little_harness.domain.steps import AgentSteps
from little_harness.domain.tool_result import ToolRunResult
from little_harness.domain.values.model_call_metrics import ModelCallMetrics
from little_harness.domain.values.numeric_values import ElapsedSeconds, Iteration
from little_harness.domain.values.text_values import (
    MessageContent,
    Prompt,
    RunId,
    SessionId,
    ToolName,
    ToolOutput,
)

from little_harness_session_jsonl.infrastructure.jsonl_appender import JsonlFileAppender
from little_harness_session_jsonl.infrastructure.jsonl_observer import (
    JsonlSessionObserver,
)


class TestJsonlSessionObserver:
    def test_logs_run_started(self, tmp_path: Path) -> None:
        file_path = tmp_path / "test.jsonl"
        appender = JsonlFileAppender(file_path)
        observer = JsonlSessionObserver(SessionId("test"), appender)

        observer.on_run_started(RunId("r1"), Prompt("hello"))

        content = json.loads(file_path.read_text())
        assert content == {
            "session_id": "test",
            "type": "run_started",
            "run_id": "r1",
            "prompt": "hello",
        }

    def test_logs_model_completed(self, tmp_path: Path) -> None:
        file_path = tmp_path / "test.jsonl"
        appender = JsonlFileAppender(file_path)
        observer = JsonlSessionObserver(SessionId("test"), appender)

        observer.on_model_completed(
            RunId("r1"), Iteration(1), MessageContent("out"), ElapsedSeconds(1.5)
        )

        content = json.loads(file_path.read_text())
        assert content == {
            "session_id": "test",
            "type": "model_completed",
            "run_id": "r1",
            "iteration": 1,
            "output": "out",
            "elapsed": 1.5,
        }

    def test_logs_model_metrics(self, tmp_path: Path) -> None:
        file_path = tmp_path / "test.jsonl"
        appender = JsonlFileAppender(file_path)
        observer = JsonlSessionObserver(SessionId("test"), appender)

        observer.on_model_metrics(
            RunId("r1"),
            Iteration(1),
            ModelCallMetrics(ElapsedSeconds(2.0), ElapsedSeconds(0.4), 20),
        )

        content = json.loads(file_path.read_text())
        assert content == {
            "session_id": "test",
            "type": "model_metrics",
            "run_id": "r1",
            "iteration": 1,
            "time_to_first_token": 0.4,
            "output_tokens": 20,
            "tokens_per_second": 10.0,
            "elapsed": 2.0,
        }

    def test_logs_decision_parsed(self, tmp_path: Path) -> None:
        file_path = tmp_path / "test.jsonl"
        appender = JsonlFileAppender(file_path)
        observer = JsonlSessionObserver(SessionId("test"), appender)

        decision = FinalAnswer(MessageContent("done"))
        observer.on_decision_parsed(RunId("r1"), Iteration(1), decision)

        content = json.loads(file_path.read_text())
        assert content == {
            "session_id": "test",
            "type": "decision_parsed",
            "run_id": "r1",
            "iteration": 1,
            "decision": "FinalAnswer",
        }

    def test_logs_tool_invoked(self, tmp_path: Path) -> None:
        file_path = tmp_path / "test.jsonl"
        appender = JsonlFileAppender(file_path)
        observer = JsonlSessionObserver(SessionId("test"), appender)

        result = ToolRunResult(
            ToolName("test_tool"), ToolOutput("data"), succeeded=True
        )
        observer.on_tool_invoked(RunId("r1"), Iteration(1), result, ElapsedSeconds(0.1))

        content = json.loads(file_path.read_text())
        assert content == {
            "session_id": "test",
            "type": "tool_invoked",
            "run_id": "r1",
            "iteration": 1,
            "tool_name": "test_tool",
            "output": "data",
            "succeeded": True,
            "elapsed": 0.1,
        }

    def test_logs_repair(self, tmp_path: Path) -> None:
        file_path = tmp_path / "test.jsonl"
        appender = JsonlFileAppender(file_path)
        observer = JsonlSessionObserver(SessionId("test"), appender)

        error = AgentProtocolError("bad format")
        observer.on_repair(RunId("r1"), Iteration(1), error)

        content = json.loads(file_path.read_text())
        assert content == {
            "session_id": "test",
            "type": "repair",
            "run_id": "r1",
            "iteration": 1,
            "error": "AgentProtocolError",
            "message": "bad format",
        }

    def test_logs_run_finished(self, tmp_path: Path) -> None:
        file_path = tmp_path / "test.jsonl"
        appender = JsonlFileAppender(file_path)
        observer = JsonlSessionObserver(SessionId("test"), appender)

        result = AgentResult(MessageContent("done"), ElapsedSeconds(2.0), AgentSteps())
        observer.on_run_finished(RunId("r1"), result)

        content = json.loads(file_path.read_text())
        assert content == {
            "session_id": "test",
            "type": "run_finished",
            "run_id": "r1",
            "answer": "done",
            "elapsed": 2.0,
        }
