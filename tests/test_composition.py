from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from llama_cpp.llama_types import CreateChatCompletionResponse

from local_llm.composition import build_application, run_cli
from local_llm.domain.values.text_values import Prompt
from local_llm.presentation.cli.app_config import AppConfig
from local_llm.presentation.cli.argument_parser import ArgumentParser
from tests.application.fakes import RecordingObserver


class JsonReplyLlama:
    """Fake llama model that always returns one valid final-answer JSON object."""

    def __init__(self, **_: Any) -> None:
        self._reply = (
            '{"action":"final","tool_name":null,"tool_input":null,'
            '"answer":"hello from the agent"}'
        )

    def create_chat_completion(self, **_: Any) -> CreateChatCompletionResponse:
        return cast(
            "CreateChatCompletionResponse",
            {"choices": [{"message": {"content": self._reply}}]},
        )


@pytest.fixture
def model_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "model.gguf"
    path.write_bytes(b"")
    monkeypatch.setattr(
        "local_llm.infrastructure.llama_cpp.model_factory.Llama", JsonReplyLlama
    )
    return path


class TestComposition:
    def test_run_cli_wires_the_whole_stack_and_renders_the_answer(
        self,
        model_file: Path,
    ) -> None:
        # Act
        output = run_cli(["--model-path", str(model_file), "--prompt", "hi"])

        # Assert
        assert "hello from the agent" in output
        assert "Elapsed:" in output

    def test_build_application_threads_the_observer_through_the_stack(
        self,
        model_file: Path,
    ) -> None:
        # Arrange
        observer = RecordingObserver()
        config = build_config(model_file)

        # Act
        build_application(config, observer).run(Prompt("hi"))

        # Assert: the seam reaches a real run end-to-end.
        assert observer.events[0] == "run_started:hi"
        assert observer.events[-1] == "run_finished"
        assert len(observer.finished) == 1


def build_config(model_file: Path) -> AppConfig:
    return ArgumentParser().parse(["--model-path", str(model_file), "--prompt", "hi"])
