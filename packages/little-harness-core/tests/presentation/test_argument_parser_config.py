from __future__ import annotations

from little_harness.domain.values.numeric_values import (
    MaxIterations,
    MaxTokens,
    RepeatPenalty,
    Temperature,
    TopP,
)
from little_harness.infrastructure.config.config_types import Config
from little_harness.presentation.cli.argument_parser import ArgumentParser


class TestArgumentParserWithConfig:
    def test_uses_config_values_as_defaults(self) -> None:
        config = Config(temperature=0.7, max_tokens=4096, max_iterations=10)
        result = ArgumentParser(config).parse([])

        assert result.temperature == Temperature(0.7)
        assert result.max_tokens == MaxTokens(4096)
        assert result.max_iterations == MaxIterations(10)

    def test_cli_overrides_config_values(self) -> None:
        config = Config(temperature=0.7, max_tokens=4096)
        result = ArgumentParser(config).parse(
            ["--temperature", "0.5", "--max-tokens", "512"]
        )

        assert result.temperature == Temperature(0.5)
        assert result.max_tokens == MaxTokens(512)

    def test_config_model_sets_provider_options(self) -> None:
        config = Config(model="gpt-4o")
        result = ArgumentParser(config).parse([])

        assert result.provider_options == {"model": "gpt-4o"}

    def test_cli_model_overrides_config_model(self) -> None:
        config = Config(model="gpt-4o")
        result = ArgumentParser(config).parse(["--model", "claude-3"])

        assert result.provider_options == {"model": "claude-3"}

    def test_config_observer_sets_observer_name(self) -> None:
        config = Config(observer="logging")
        result = ArgumentParser(config).parse([])

        assert result.observer_name == "logging"

    def test_cli_observer_overrides_config_observer(self) -> None:
        config = Config(observer="logging")
        result = ArgumentParser(config).parse(["--observer", "otel"])

        assert result.observer_name == "otel"

    def test_config_stream_sets_enable_streaming(self) -> None:
        config = Config(stream=True)
        result = ArgumentParser(config).parse([])

        assert result.enable_streaming is True

    def test_config_tools_sets_tool_selection(self) -> None:
        config = Config(tools=("read_file", "bash"))
        result = ArgumentParser(config).parse([])

        assert result.tool_selection == ("read_file", "bash")

    def test_cli_tools_overrides_config_tools(self) -> None:
        config = Config(tools=("read_file",))
        result = ArgumentParser(config).parse(["--tools", "calculator"])

        assert result.tool_selection == ("calculator",)

    def test_config_yes_sets_approve_all(self) -> None:
        config = Config(approve_all=True)
        result = ArgumentParser(config).parse([])

        assert result.approve_all is True

    def test_config_provider_sets_provider(self) -> None:
        config = Config(provider="litellm")
        result = ArgumentParser(config).parse([])

        assert result.provider == "litellm"

    def test_config_policy_sets_policy(self) -> None:
        config = Config(policy="json")
        result = ArgumentParser(config).parse([])

        assert result.policy == "json"

    def test_config_top_p_sets_top_p(self) -> None:
        config = Config(top_p=0.9)
        result = ArgumentParser(config).parse([])

        assert result.top_p == TopP(0.9)

    def test_config_repeat_penalty_sets_repeat_penalty(self) -> None:
        config = Config(repeat_penalty=1.1)
        result = ArgumentParser(config).parse([])

        assert result.repeat_penalty == RepeatPenalty(1.1)

    def test_config_ui_sets_ui(self) -> None:
        config = Config(ui="rich")
        result = ArgumentParser(config).parse([])

        assert result.ui == "rich"

    def test_config_plugins_available_as_plugin_configs(self) -> None:
        config = Config(plugins={"llama_cpp": {"n_ctx": "8192"}})
        result = ArgumentParser(config).parse([])

        assert result.plugin_configs == {"llama_cpp": {"n_ctx": "8192"}}

    def test_profile_cli_arg_is_read(self) -> None:
        result = ArgumentParser().parse(["--profile", "fast"])

        assert result.profile == "fast"

    def test_profile_defaults_to_none(self) -> None:
        result = ArgumentParser().parse([])

        assert result.profile is None

    def test_code_defaults_still_apply_when_no_config(self) -> None:
        result = ArgumentParser().parse([])

        assert result.temperature == Temperature(0.1)
        assert result.max_tokens == MaxTokens(512)
        assert result.max_iterations == MaxIterations(5)
        assert result.ui == "default"
