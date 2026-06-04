from __future__ import annotations

import pytest
from little_harness.domain.values.numeric_values import (
    MaxIterations,
    MaxTokens,
    Temperature,
)
from little_harness.domain.values.text_values import Prompt
from little_harness.presentation.cli.app_config import AppConfig
from little_harness.presentation.cli.argument_parser import (
    ArgumentParser,
    build_parser,
)

DEFAULT_PROMPT_TEXT = (
    "What is 144 divided by 12? Then tell me if the result is even or odd."
)

EXPECTED_ARGUMENTS: list[tuple[str, list[str], str, object]] = [
    ("prompt", ["-p", "--prompt"], "Prompt to send to the model.", None),
    ("temperature", ["--temperature"], "Sampling temperature.", 0.0),
    ("max_tokens", ["--max-tokens"], "Maximum generated tokens.", 512),
    ("max_iterations", ["--max-iterations"], "Maximum agent loop iterations.", 5),
    ("log", ["--log"], "Emit structured JSON logs for each agent event.", False),
    (
        "stream",
        ["--stream"],
        "Stream generated tokens to stdout as they are produced.",
        False,
    ),
    (
        "provider",
        ["--provider"],
        "Chat model provider to use (an installed plugin name). "
        "Defaults to the sole installed provider.",
        None,
    ),
    (
        "model",
        ["-m", "--model"],
        "Model to use; provider-specific (a model name for litellm, a GGUF path "
        "for llama_cpp). Shorthand for -o model=...",
        None,
    ),
    (
        "options",
        ["-o", "--option"],
        "Provider-specific setting, repeatable (e.g. -o n_ctx=8192).",
        None,
    ),
    (
        "tools",
        ["--tools"],
        "Comma-separated tool names to enable (installed plugin names). "
        "Defaults to every installed tool.",
        None,
    ),
    (
        "approve_all",
        ["--yes"],
        "Approve every sensitive tool without prompting (non-interactive runs).",
        False,
    ),
]


class TestBuildParser:
    def test_describes_the_program(self) -> None:
        assert build_parser().description == "Run a small local LLM agent."

    @pytest.mark.parametrize(
        ("dest", "option_strings", "help_text", "default"),
        EXPECTED_ARGUMENTS,
    )
    def test_each_argument_is_configured_exactly(
        self,
        dest: str,
        option_strings: list[str],
        help_text: str,
        default: object,
    ) -> None:
        # Arrange
        actions = {action.dest: action for action in build_parser()._actions}

        # Assert: option strings and help are pinned so wording/flags can't drift.
        action = actions[dest]
        assert action.option_strings == option_strings
        assert action.help == help_text
        if default is not None:
            assert action.default == default

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
            prompt=Prompt(DEFAULT_PROMPT_TEXT),
            temperature=Temperature(0.0),
            max_tokens=MaxTokens(512),
            max_iterations=MaxIterations(5),
            provider=None,
            provider_options={},
            enable_logging=False,
            enable_streaming=False,
        )

    def test_enables_logging_with_the_log_flag(self) -> None:
        # Act / Assert
        assert ArgumentParser().parse(["--log"]).enable_logging is True

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
