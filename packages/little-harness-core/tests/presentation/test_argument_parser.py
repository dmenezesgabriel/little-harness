from __future__ import annotations

import pytest
from little_harness.domain.values.numeric_values import (
    MaxIterations,
    MaxTokens,
    RepeatPenalty,
    Temperature,
    TopP,
)
from little_harness.domain.values.text_values import Prompt
from little_harness.presentation.cli.app_config import AppConfig
from little_harness.presentation.cli.argument_parser import (
    ArgumentParser,
    build_parser,
)

EXPECTED_ARGUMENTS: list[tuple[str, list[str], str, object, type | None]] = [
    (
        "prompt",
        ["-p", "--prompt"],
        "Prompt to send to the model.",
        None,
        None,
    ),
    (
        "temperature",
        ["--temperature"],
        "Sampling temperature. Default: 0.1.",
        None,
        float,
    ),
    (
        "top_p",
        ["--top-p"],
        "Nucleus sampling threshold (0.0..1.0). Default lets provider decide.",
        None,
        float,
    ),
    (
        "repeat_penalty",
        ["--repeat-penalty"],
        "Repetition penalty (0.0..2.0, 1.0 = off). Provider default when unset.",
        None,
        float,
    ),
    (
        "max_tokens",
        ["--max-tokens"],
        "Maximum generated tokens. Default: 512.",
        None,
        int,
    ),
    (
        "max_iterations",
        ["--max-iterations"],
        "Maximum agent loop iterations. Default: 5.",
        None,
        int,
    ),
    ("log", ["--log"], "Shorthand for --observer logging.", False, None),
    (
        "observer",
        ["--observer"],
        "Observer plugin to use (an installed plugin name, e.g. 'logging'). "
        "Defaults to no observer.",
        None,
        None,
    ),
    (
        "stream",
        ["--stream"],
        "Stream generated tokens to stdout as they are produced.",
        False,
        None,
    ),
    (
        "provider",
        ["--provider"],
        "Chat model provider to use (an installed plugin name). "
        "Defaults to the sole installed provider.",
        None,
        None,
    ),
    (
        "policy",
        ["--policy"],
        "Agent policy plugin to use (an installed plugin name, e.g. 'json'). "
        "Defaults to the sole installed policy.",
        None,
        None,
    ),
    (
        "profile",
        ["--profile"],
        "Profile to activate (overrides config default).",
        None,
        None,
    ),
    (
        "model",
        ["-m", "--model"],
        "Model to use; provider-specific (a model name for litellm, a GGUF path "
        "for llama_cpp). Shorthand for -o model=...",
        None,
        None,
    ),
    (
        "options",
        ["-o", "--option"],
        "Provider-specific setting, repeatable (e.g. -o n_ctx=8192).",
        [],
        None,
    ),
    (
        "tools",
        ["--tools"],
        "Comma-separated tool names to enable (installed plugin names). "
        "Defaults to every installed tool.",
        None,
        None,
    ),
    (
        "approve_all",
        ["--yes"],
        "Approve every sensitive tool without prompting (non-interactive runs).",
        False,
        None,
    ),
    (
        "ui",
        ["--ui"],
        "Interactive UI plugin to use (e.g. 'rich', 'default'). Default: default.",
        None,
        None,
    ),
]


class TestBuildParser:
    def test_describes_the_program(self) -> None:
        assert build_parser().description == "Run a small local LLM agent."

    @pytest.mark.parametrize(
        ("dest", "option_strings", "help_text", "default", "expected_type"),
        EXPECTED_ARGUMENTS,
    )
    def test_each_argument_is_configured_exactly(
        self,
        dest: str,
        option_strings: list[str],
        help_text: str,
        default: object,
        expected_type: type | None,
    ) -> None:
        # Arrange
        actions = {action.dest: action for action in build_parser()._actions}

        # Assert: option strings and help are pinned so wording/flags can't drift.
        action = actions[dest]
        assert action.option_strings == option_strings
        assert action.help == help_text
        assert action.default == default
        assert action.type is expected_type

    def test_option_flag_uses_a_key_value_metavar(self) -> None:
        # Assert: the metavar documents the expected -o argument shape.
        actions = {action.dest: action for action in build_parser()._actions}
        assert actions["options"].metavar == "KEY=VALUE"


class TestArgumentParser:
    def test_uses_defaults_when_no_flags_given(self) -> None:
        # Act
        config = ArgumentParser().parse([])

        # Assert
        assert config == AppConfig(
            prompt=None,
            temperature=Temperature(0.1),
            max_tokens=MaxTokens(512),
            max_iterations=MaxIterations(5),
            provider=None,
            provider_options={},
            enable_streaming=False,
            ui="default",
        )

    def test_reads_the_ui_override(self) -> None:
        # Act / Assert
        assert ArgumentParser().parse(["--ui", "rich"]).ui == "rich"

    def test_log_flag_is_shorthand_for_the_logging_observer(self) -> None:
        # Act / Assert
        assert ArgumentParser().parse(["--log"]).observer_name == "logging"

    def test_explicit_observer_overrides_the_log_shorthand(self) -> None:
        # Act / Assert
        config = ArgumentParser().parse(["--observer", "otel", "--log"])
        assert config.observer_name == "otel"

    def test_defaults_observer_name_to_none(self) -> None:
        # Act / Assert
        assert ArgumentParser().parse([]).observer_name is None

    def test_reads_the_policy_override(self) -> None:
        # Act / Assert
        assert ArgumentParser().parse(["--policy", "json"]).policy == "json"

    def test_enables_streaming_with_the_stream_flag(self) -> None:
        # Act / Assert
        assert ArgumentParser().parse(["--stream"]).enable_streaming is True

    def test_reads_the_provider_override(self) -> None:
        # Act / Assert
        assert ArgumentParser().parse(["--provider", "litellm"]).provider == "litellm"

    def test_collects_repeated_provider_options(self) -> None:
        # Act
        config = ArgumentParser().parse(
            ["-o", "model_path=/tmp/m.gguf", "--option", "n_ctx=4096"]
        )

        # Assert
        assert config.provider_options == {
            "model_path": "/tmp/m.gguf",
            "n_ctx": "4096",
        }

    def test_model_flag_sets_the_model_option(self) -> None:
        # Act / Assert: --model is shorthand for -o model=...
        config = ArgumentParser().parse(["--model", "gemini/gemini-2.5-flash"])
        assert config.provider_options == {"model": "gemini/gemini-2.5-flash"}

    def test_explicit_model_option_overrides_the_model_flag(self) -> None:
        # Act
        config = ArgumentParser().parse(["--model", "a", "-o", "model=b"])

        # Assert
        assert config.provider_options == {"model": "b"}

    def test_keeps_equals_signs_inside_an_option_value(self) -> None:
        # Act
        config = ArgumentParser().parse(["-o", "api_base=https://x/y?a=b"])

        # Assert
        assert config.provider_options == {"api_base": "https://x/y?a=b"}

    def test_reads_runtime_overrides_from_argv(self) -> None:
        # Act
        config = ArgumentParser().parse(
            [
                "--prompt",
                "What is 2 + 2?",
                "--temperature",
                "0.7",
                "--top-p",
                "0.5",
                "--repeat-penalty",
                "1.1",
                "--max-tokens",
                "256",
                "--max-iterations",
                "3",
            ]
        )

        # Assert
        assert config == AppConfig(
            prompt=Prompt("What is 2 + 2?"),
            temperature=Temperature(0.7),
            top_p=TopP(0.5),
            repeat_penalty=RepeatPenalty(1.1),
            max_tokens=MaxTokens(256),
            max_iterations=MaxIterations(3),
        )

    def test_rejects_invalid_runtime_value_at_the_boundary(self) -> None:
        # Act / Assert: a zero token budget fails when the value object is built.
        with pytest.raises(ValueError, match="MaxTokens is not positive"):
            ArgumentParser().parse(["--max-tokens", "0"])

    def test_rejects_a_malformed_option_pair(self) -> None:
        # Act / Assert: the message names the offending value and expected shape.
        with pytest.raises(ValueError, match="Invalid --option: 'no-separator'"):
            ArgumentParser().parse(["-o", "no-separator"])

    def test_rejects_an_option_with_an_empty_key(self) -> None:
        # Act / Assert: a leading '=' yields an empty key and is rejected.
        with pytest.raises(ValueError, match="Invalid --option: '=value'"):
            ArgumentParser().parse(["-o", "=value"])

    def test_defaults_tool_selection_to_none(self) -> None:
        # Act / Assert: omitting --tools means every installed tool.
        assert ArgumentParser().parse([]).tool_selection is None

    def test_selects_tools_from_a_comma_separated_list(self) -> None:
        # Act
        config = ArgumentParser().parse(["--tools", "read_file,bash"])

        # Assert
        assert config.tool_selection == ("read_file", "bash")

    def test_trims_whitespace_around_each_tool_name(self) -> None:
        # Act
        config = ArgumentParser().parse(["--tools", " read_file , bash "])

        # Assert
        assert config.tool_selection == ("read_file", "bash")

    def test_rejects_an_empty_tool_name(self) -> None:
        # Act / Assert: a trailing comma yields an empty name and is rejected.
        with pytest.raises(ValueError, match="Invalid --tools: 'read_file,'"):
            ArgumentParser().parse(["--tools", "read_file,"])

    def test_enables_approve_all_with_the_yes_flag(self) -> None:
        # Act / Assert
        assert ArgumentParser().parse(["--yes"]).approve_all is True
