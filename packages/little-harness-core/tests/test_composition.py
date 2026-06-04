from __future__ import annotations

import logging
from collections.abc import Mapping

import pytest
from little_harness.application.agent_runtime import AgentRuntimeConfig
from little_harness.application.ports.chat_model import ChatModel
from little_harness.composition import (
    build_application,
    build_chat_model,
    build_hooks,
    build_observer,
    build_token_sink,
    run_cli,
    to_runtime_config,
)
from little_harness.domain.errors import UnknownProviderError
from little_harness.domain.values.numeric_values import (
    MaxIterations,
    MaxTokens,
    Temperature,
)
from little_harness.domain.values.text_values import Prompt, RunId
from little_harness.infrastructure.hooks.null_hook import NullHook
from little_harness.infrastructure.observability.null_observer import NullObserver
from little_harness.infrastructure.observability.structured_logging_observer import (
    StructuredLoggingObserver,
)
from little_harness.plugin_discovery import PROVIDER_GROUP
from little_harness.presentation.cli.argument_parser import ArgumentParser
from little_harness.presentation.cli.token_sinks import NullTokenSink, StdoutTokenSink

from tests.application.fakes import RecordingObserver
from tests.plugin_fakes import (
    FakeChatModel,
    FakeEntryPoint,
    install_entry_points,
    make_provider_builder,
)

FINAL_ANSWER_JSON = (
    '{"action":"final","tool_name":null,"tool_input":null,'
    '"answer":"hello from the agent"}'
)


@pytest.fixture
def created_models(monkeypatch: pytest.MonkeyPatch) -> list[FakeChatModel]:
    """Register a fake `llama_cpp` provider and capture each model it builds."""
    created: list[FakeChatModel] = []

    def build(_options: Mapping[str, str]) -> ChatModel:
        model = FakeChatModel(FINAL_ANSWER_JSON)
        created.append(model)
        return model

    install_entry_points(
        monkeypatch, {PROVIDER_GROUP: [FakeEntryPoint("llama_cpp", build)]}
    )
    return created


@pytest.mark.usefixtures("created_models")
class TestComposition:
    def test_run_cli_wires_the_whole_stack_and_renders_the_answer(self) -> None:
        # Act
        output = run_cli(["--prompt", "hi"])

        # Assert
        assert "hello from the agent" in output
        assert "Elapsed:" in output

    def test_build_application_threads_the_observer_through_the_stack(self) -> None:
        # Arrange
        observer = RecordingObserver()
        config = ArgumentParser().parse(["--prompt", "hi"])

        # Act
        build_application(config, observer).run(Prompt("hi"))

        # Assert: the seam reaches a real run end-to-end.
        assert observer.events[0] == "run_started:hi"
        assert observer.events[-1] == "run_finished"
        assert len(observer.finished) == 1

    def test_run_cli_with_log_flag_emits_lifecycle_logs(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Act: --log must wire the structured observer, not the null one.
        with caplog.at_level(logging.INFO):
            run_cli(["--prompt", "hi", "--log"])

        # Assert
        messages = [record.getMessage() for record in caplog.records]
        assert any('"event": "run_started"' in message for message in messages)


class TestModelLifecycle:
    def test_run_cli_closes_the_model_after_the_run(
        self, created_models: list[FakeChatModel]
    ) -> None:
        # Act
        run_cli(["--prompt", "hi"])

        # Assert
        assert created_models and created_models[0].closed is True


class TestProviderSelection:
    def test_builds_the_selected_provider_via_discovery(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        install_entry_points(
            monkeypatch,
            {PROVIDER_GROUP: [FakeEntryPoint("llama_cpp", make_provider_builder(""))]},
        )
        config = ArgumentParser().parse(["--prompt", "hi"])

        # Act / Assert
        assert isinstance(build_chat_model(config), FakeChatModel)

    def test_rejects_no_selection_when_several_providers_installed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange: two providers, and the config names none (provider stays None).
        install_entry_points(
            monkeypatch,
            {
                PROVIDER_GROUP: [
                    FakeEntryPoint("litellm", make_provider_builder("")),
                    FakeEntryPoint("llama_cpp", make_provider_builder("")),
                ]
            },
        )
        config = ArgumentParser().parse(["--prompt", "hi"])

        # Act / Assert: ambiguous default fails instead of guessing.
        with pytest.raises(UnknownProviderError, match=r"2 installed"):
            build_chat_model(config)

    def test_rejects_an_unknown_provider_with_the_installed_list(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        install_entry_points(
            monkeypatch,
            {PROVIDER_GROUP: [FakeEntryPoint("llama_cpp", make_provider_builder(""))]},
        )
        config = ArgumentParser().parse(["--prompt", "hi", "--provider", "mystery"])

        # Act / Assert
        with pytest.raises(
            UnknownProviderError, match=r"Unknown provider: 'mystery'.*\['llama_cpp'\]"
        ):
            build_chat_model(config)

    def test_passes_provider_options_to_the_builder(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange: capture the options the selected builder receives.
        received: list[Mapping[str, str]] = []

        def build(options: Mapping[str, str]) -> ChatModel:
            received.append(options)
            return FakeChatModel("")

        install_entry_points(
            monkeypatch, {PROVIDER_GROUP: [FakeEntryPoint("llama_cpp", build)]}
        )
        config = ArgumentParser().parse(["--prompt", "hi", "-o", "model_path=/m.gguf"])

        # Act
        build_chat_model(config)

        # Assert
        assert received == [{"model_path": "/m.gguf"}]


class TestRuntimeConfig:
    def test_copies_the_sampling_and_loop_bounds(self) -> None:
        # Arrange
        config = ArgumentParser().parse(
            [
                "--prompt",
                "hi",
                "--temperature",
                "0.5",
                "--max-tokens",
                "99",
                "--max-iterations",
                "7",
            ]
        )

        # Act / Assert
        assert to_runtime_config(config) == AgentRuntimeConfig(
            max_iterations=MaxIterations(7),
            temperature=Temperature(0.5),
            max_tokens=MaxTokens(99),
        )


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

    def test_structured_observer_logs_under_the_agent_logger(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Arrange
        config = ArgumentParser().parse(["--prompt", "hi", "--log"])
        observer = build_observer(config)

        # Act
        with caplog.at_level(logging.INFO):
            observer.on_run_started(RunId("rid"), Prompt("hi"))

        # Assert: the real logger is named "agent" and carries the payload.
        record = caplog.records[-1]
        assert record.name == "agent"
        assert '"run_id": "rid"' in record.getMessage()


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


class TestHookSelection:
    def test_defaults_to_the_null_hook(self) -> None:
        # Act / Assert: no hooks are configured by default.
        assert isinstance(build_hooks(), NullHook)
