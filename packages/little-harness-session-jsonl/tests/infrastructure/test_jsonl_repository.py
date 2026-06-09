# ruff: noqa: D100, D101, D102, D103
import json
from pathlib import Path
from typing import Any

from little_harness.domain.decision import AgentDecision
from little_harness.domain.message import ChatMessage
from little_harness.domain.tool_result import ToolRunResult
from little_harness.domain.values.role import ASSISTANT, SYSTEM, USER
from little_harness.domain.values.text_values import MessageContent, Prompt, SessionId

from little_harness_session_jsonl.infrastructure.jsonl_repository import (
    JsonlSessionRepository,
)


class FakeAgentPolicy:
    def build_tool_observation_message(
        self, original_prompt: Prompt, tool_result: ToolRunResult
    ) -> ChatMessage:
        content = (
            f"Observation: {tool_result.output.value} "
            f"({tool_result.tool_name.value}, {tool_result.succeeded}, {original_prompt.value})"
        )
        return ChatMessage(SYSTEM, MessageContent(content))

    def build_repair_message(
        self, original_prompt: Prompt, error: Exception
    ) -> ChatMessage:
        content = f"Repair: {error} ({original_prompt.value})"
        return ChatMessage(SYSTEM, MessageContent(content))

    # We don't need the other methods for the repository reconstruction
    def system_prompt(self, tools: Any) -> MessageContent:
        del tools
        return MessageContent("")

    def response_schema(self, tools: Any) -> Any | None:
        del tools
        return None

    def parse_model_output(self, output: MessageContent) -> AgentDecision: ...


class TestJsonlSessionRepository:
    def test_loads_empty_history_when_file_does_not_exist(self, tmp_path: Path) -> None:
        repo = JsonlSessionRepository(tmp_path, FakeAgentPolicy())

        history = repo.load(SessionId("missing"))

        assert len(history) == 0

    def test_reconstructs_history_from_jsonl_events(self, tmp_path: Path) -> None:
        file_path = tmp_path / "test.jsonl"
        events = [
            {
                "session_id": "test",
                "type": "run_started",
                "run_id": "r1",
                "prompt": "hello",
            },
            {
                "session_id": "test",
                "type": "model_completed",
                "run_id": "r1",
                "iteration": 1,
                "output": "thinking...",
                "elapsed": 1.0,
            },
            {
                "session_id": "test",
                "type": "tool_invoked",
                "run_id": "r1",
                "iteration": 1,
                "tool_name": "calc",
                "output": "4",
                "succeeded": True,
                "elapsed": 0.5,
            },
            {
                "session_id": "test",
                "type": "model_completed",
                "run_id": "r1",
                "iteration": 2,
                "output": "bad JSON",
                "elapsed": 1.0,
            },
            {
                "session_id": "test",
                "type": "repair",
                "run_id": "r1",
                "iteration": 2,
                "error": "AgentProtocolError",
                "message": "bad format",
            },
        ]
        with file_path.open("w") as f:
            for event in events:
                f.write(json.dumps(event) + "\n")

        repo = JsonlSessionRepository(tmp_path, FakeAgentPolicy())
        history = repo.load(SessionId("test"))
        messages = list(history)

        assert len(messages) == 5
        assert messages[0].role == USER
        assert messages[0].content.value == "hello"

        assert messages[1].role == ASSISTANT
        assert messages[1].content.value == "thinking..."

        assert messages[2].role == SYSTEM
        assert messages[2].content.value == "Observation: 4 (calc, True, hello)"

        assert messages[3].role == ASSISTANT
        assert messages[3].content.value == "bad JSON"

        assert messages[4].role == SYSTEM
        assert messages[4].content.value == "Repair: bad format (hello)"

    def test_ignores_orphaned_events_without_current_prompt(
        self, tmp_path: Path
    ) -> None:
        file_path = tmp_path / "test.jsonl"
        events = [
            # Missing run_started, so current_prompt is None
            {
                "session_id": "test",
                "type": "tool_invoked",
                "run_id": "r1",
                "iteration": 1,
                "tool_name": "calc",
                "output": "4",
                "succeeded": True,
            },
            {
                "session_id": "test",
                "type": "repair",
                "run_id": "r1",
                "iteration": 1,
                "error": "AgentProtocolError",
                "message": "bad",
            },
        ]
        with file_path.open("w") as f:
            for event in events:
                f.write(json.dumps(event) + "\n")
            f.write("\n")  # empty line to kill `not line.strip()` mutant
            f.write("    \n")

        repo = JsonlSessionRepository(tmp_path, FakeAgentPolicy())
        history = repo.load(SessionId("test"))
        messages = list(history)

        assert len(messages) == 0

    def test_file_is_opened_with_explicit_kwargs(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        file_path = tmp_path / "test.jsonl"
        file_path.touch()

        opened_kwargs = {}

        original_open = Path.open

        def mock_open(self: Path, *args: Any, **kwargs: Any) -> Any:
            nonlocal opened_kwargs
            if self == file_path:
                opened_kwargs = kwargs.copy()
                if args:
                    opened_kwargs["mode"] = args[0]
            return original_open(self, *args, **kwargs)

        monkeypatch.setattr(Path, "open", mock_open)

        repo = JsonlSessionRepository(tmp_path, FakeAgentPolicy())
        repo.load(SessionId("test"))

        assert opened_kwargs.get("mode") == "r"
        assert opened_kwargs.get("encoding") == "utf-8"
