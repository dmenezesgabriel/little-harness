from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest
from llama_cpp.llama_types import CreateChatCompletionStreamResponse

from local_llm.composition import (
    build_application,
    build_chat_model,
    build_observer,
    build_token_sink,
    run_cli,
)
from local_llm.domain.values.text_values import Prompt
from local_llm.infrastructure.llama_cpp.chat_model import LlamaCppChatModel
from local_llm.infrastructure.observability.null_observer import NullObserver
from local_llm.infrastructure.observability.structured_logging_observer import (
    StructuredLoggingObserver,
)
from local_llm.presentation.cli.app_config import AppConfig
from local_llm.presentation.cli.argument_parser import ArgumentParser
from local_llm.presentation.cli.token_sinks import NullTokenSink, StdoutTokenSink
from tests.application.fakes import RecordingObserver


class JsonReplyLlama:
    """Fake llama model that streams one valid final-answer JSON object."""

    def __init__(self, **_: Any) -> None:
        self._reply = (
            '{"action":"final","tool_name":null,"tool_input":null,'
            '"answer":"hello from the agent"}'
        )
        self.closed = False

    def create_chat_completion(
        self, **_: Any
    ) -> Iterator[CreateChatCompletionStreamResponse]:
        return iter(
            [
                cast(
                    "CreateChatCompletionStreamResponse",
                    {"choices": [{"delta": {"content": self._reply}}]},
                )
            ]
        )

    def close(self) -> None:
        self.closed = True


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


class TestModelLifecycle:
    def test_run_cli_closes_the_model_after_the_run(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Arrange: capture each constructed model so we can assert it was closed.
        created: list[JsonReplyLlama] = []

        def recording_llama(**kwargs: Any) -> JsonReplyLlama:
            instance = JsonReplyLlama(**kwargs)
            created.append(instance)
            return instance

        path = tmp_path / "model.gguf"
        path.write_bytes(b"")
        monkeypatch.setattr(
            "local_llm.infrastructure.llama_cpp.model_factory.Llama", recording_llama
        )

        # Act
        run_cli(["--model-path", str(path), "--prompt", "hi"])

        # Assert
        assert created and created[0].closed is True


class TestProviderSelection:
    def test_builds_the_default_llama_cpp_provider(self, model_file: Path) -> None:
        # Arrange
        config = ArgumentParser().parse(
            ["--model-path", str(model_file), "--prompt", "hi"]
        )

        # Act / Assert
        assert isinstance(build_chat_model(config), LlamaCppChatModel)

    def test_rejects_an_unknown_provider_with_the_known_list(self) -> None:
        # Arrange
        config = ArgumentParser().parse(["--prompt", "hi", "--provider", "mystery"])

        # Act / Assert
        with pytest.raises(
            ValueError, match=r"Unknown provider: 'mystery'.*\['llama_cpp'\]"
        ):
            build_chat_model(config)


class TestObserverSelection:
    def test_defaults_to_the_null_observer(self) -> None:
        # Arrange
        config = ArgumentParser().parse(["--prompt", "hi"])

        # Act / Assert
        assert isinstance(build_observer(config), NullObserver)

    def test_selects_structured_logging_when_log_flag_set(self) -> None:
        # Arrange
        config = ArgumentParser().parse(["--prompt", "hi", "--log"])

        # Act / Assert
        assert isinstance(build_observer(config), StructuredLoggingObserver)


class TestTokenSinkSelection:
    def test_defaults_to_the_null_token_sink(self) -> None:
        # Arrange
        config = ArgumentParser().parse(["--prompt", "hi"])

        # Act / Assert
        assert isinstance(build_token_sink(config), NullTokenSink)

    def test_selects_stdout_sink_when_stream_flag_set(self) -> None:
        # Arrange
        config = ArgumentParser().parse(["--prompt", "hi", "--stream"])

        # Act / Assert
        assert isinstance(build_token_sink(config), StdoutTokenSink)


def build_config(model_file: Path) -> AppConfig:
    return ArgumentParser().parse(["--model-path", str(model_file), "--prompt", "hi"])
