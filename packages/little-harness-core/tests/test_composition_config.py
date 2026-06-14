from __future__ import annotations

from collections.abc import Mapping

import pytest
from little_harness.application.ports.chat_model import ChatModel
from little_harness.composition import build_chat_model, run_cli
from little_harness.domain.values.numeric_values import (
    MaxIterations,
    MaxTokens,
    Temperature,
)
from little_harness.presentation.cli.app_config import AppConfig

from tests.plugin_fakes import (
    FakeChatModel,
    FakeEntryPoint,
    install_entry_points,
    make_policy_builder,
    make_provider_builder,
)

FINAL_ANSWER_JSON = (
    '{"action":"final","tool_name":null,"tool_input":null,'
    '"answer":"hello from the agent"}'
)


@pytest.fixture
def fake_provider_and_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    install_entry_points(
        monkeypatch,
        {
            "little_harness.chat_model_providers": [
                FakeEntryPoint("llama_cpp", make_provider_builder(FINAL_ANSWER_JSON))
            ],
            "little_harness.agent_policies": [
                FakeEntryPoint("json", make_policy_builder())
            ],
        },
    )


class TestBuildChatModelWithPluginConfig:
    def test_plugin_config_and_cli_options_are_merged(
        self, fake_provider_and_policy: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CLI --option values override matching plugin config keys."""
        received: list[Mapping[str, str]] = []

        def capturing_build(options: Mapping[str, str]) -> ChatModel:
            received.append(options)
            return FakeChatModel(FINAL_ANSWER_JSON)

        install_entry_points(
            monkeypatch,
            {
                "little_harness.chat_model_providers": [
                    FakeEntryPoint("llama_cpp", capturing_build)
                ],
                "little_harness.agent_policies": [
                    FakeEntryPoint("json", make_policy_builder())
                ],
            },
        )

        base = AppConfig(
            temperature=Temperature(0.1),
            max_tokens=MaxTokens(512),
            max_iterations=MaxIterations(5),
            provider="llama_cpp",
            provider_options={"n_ctx": "4096"},
            plugin_configs={"llama_cpp": {"model": "gpt-4o", "n_ctx": "8192"}},
        )

        build_chat_model(base)

        assert len(received) == 1
        opts = received[0]
        # CLI n_ctx wins over plugin config
        assert opts["n_ctx"] == "4096"
        # Plugin-only keys are present
        assert opts["model"] == "gpt-4o"

    def test_no_plugin_config_passes_cli_options_unchanged(
        self, fake_provider_and_policy: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        received: list[Mapping[str, str]] = []

        def capturing_build(options: Mapping[str, str]) -> ChatModel:
            received.append(options)
            return FakeChatModel(FINAL_ANSWER_JSON)

        install_entry_points(
            monkeypatch,
            {
                "little_harness.chat_model_providers": [
                    FakeEntryPoint("llama_cpp", capturing_build)
                ],
                "little_harness.agent_policies": [
                    FakeEntryPoint("json", make_policy_builder())
                ],
            },
        )

        base = AppConfig(
            temperature=Temperature(0.1),
            max_tokens=MaxTokens(512),
            max_iterations=MaxIterations(5),
            provider="llama_cpp",
            provider_options={"model_path": "/models/m.gguf"},
            plugin_configs={},
        )

        build_chat_model(base)

        assert len(received) == 1
        assert received[0] == {"model_path": "/models/m.gguf"}


class TestRunCliConfigIntegration:
    def test_run_cli_still_works_without_config_files(
        self, fake_provider_and_policy: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Existing behavior unchanged when no TOML files exist."""
        output = run_cli(["--prompt", "hi"])
        assert "hello from the agent" in output

    def test_profile_is_set_on_app_config(
        self, fake_provider_and_policy: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """--profile flow sets AppConfig.profile."""
        captured: list[AppConfig] = []

        def fake_build_app(cfg: AppConfig, _obs: object) -> object:
            captured.append(cfg)

            class FakeApp:
                def __enter__(self) -> FakeApp:
                    return self

                def __exit__(self, *_: object) -> None:
                    pass

                def run(self, *_: object) -> str:
                    return "rendered"

            return FakeApp()

        def fake_build_obs(_cfg: object) -> object:
            return None

        monkeypatch.setattr(
            "little_harness.composition.build_application", fake_build_app
        )
        monkeypatch.setattr("little_harness.composition.build_observer", fake_build_obs)

        output = run_cli(["--prompt", "hi", "--profile", "fast"])

        assert output == "rendered"
        assert len(captured) == 1
        assert captured[0].profile == "fast"

    def test_profile_resolved_from_config_then_overridden_by_cli(
        self, fake_provider_and_policy: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CLI --profile overrides config default profile."""
        captured: list[AppConfig] = []

        def fake_build_app(cfg: AppConfig, _obs: object) -> object:
            captured.append(cfg)

            class FakeApp:
                def __enter__(self) -> FakeApp:
                    return self

                def __exit__(self, *_: object) -> None:
                    pass

                def run(self, *_: object) -> str:
                    return "rendered"

            return FakeApp()

        def fake_build_obs(_cfg: object) -> object:
            return None

        monkeypatch.setattr(
            "little_harness.composition.build_application", fake_build_app
        )
        monkeypatch.setattr("little_harness.composition.build_observer", fake_build_obs)

        run_cli(["--prompt", "hi", "--profile", "explicit"])
        assert captured[0].profile == "explicit"
