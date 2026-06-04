from __future__ import annotations

from pathlib import Path

import pytest

from local_llm.domain.values.model_path import ModelPath
from local_llm.domain.values.numeric_values import (
    ContextSize,
    GpuLayerCount,
    MaxIterations,
    MaxTokens,
    Temperature,
    ThreadCount,
)
from local_llm.domain.values.text_values import Prompt
from local_llm.presentation.cli.app_config import AppConfig
from local_llm.presentation.cli.argument_parser import ArgumentParser, build_parser

DEFAULT_PROMPT_TEXT = (
    "Explain llama.cpp in exactly 3 short bullet points. "
    "Be specific: mention GGUF models, local inference, "
    "and CPU-friendly execution."
)

EXPECTED_ARGUMENTS: list[tuple[str, list[str], str, object]] = [
    ("prompt", ["-p", "--prompt"], "Prompt to send to the local model.", None),
    ("model_path", ["--model-path"], "Path to the local GGUF model.", None),
    ("ctx", ["--ctx"], "Context size.", 8192),
    ("threads", ["--threads"], "CPU thread count.", 8),
    ("gpu_layers", ["--gpu-layers"], "Number of GPU layers. Use 0 for CPU-only.", 0),
    ("temperature", ["--temperature"], "Sampling temperature.", 0.0),
    ("max_tokens", ["--max-tokens"], "Maximum generated tokens.", 512),
    (
        "max_iterations",
        ["--max-iterations"],
        "Maximum agent loop iterations.",
        5,
    ),
    ("log", ["--log"], "Emit structured JSON logs for each agent event.", False),
    (
        "stream",
        ["--stream"],
        "Stream generated tokens to stdout as they are produced.",
        False,
    ),
    ("provider", ["--provider"], "Chat model provider to use.", "llama_cpp"),
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


class TestArgumentParser:
    def test_uses_defaults_when_no_flags_given(self) -> None:
        # Act
        config = ArgumentParser().parse([])

        # Assert
        assert config.prompt == Prompt(DEFAULT_PROMPT_TEXT)
        assert config.model_path == ModelPath(Path("models/LFM2-8B-A1B-Q4_K_M.gguf"))
        assert config.context_size == ContextSize(8192)
        assert config.thread_count == ThreadCount(8)
        assert config.gpu_layer_count == GpuLayerCount(0)
        assert config.temperature == Temperature(0.0)
        assert config.max_tokens == MaxTokens(512)
        assert config.max_iterations == MaxIterations(5)
        assert config.enable_logging is False
        assert config.enable_streaming is False
        assert config.provider == "llama_cpp"

    def test_enables_logging_with_the_log_flag(self) -> None:
        # Act
        config = ArgumentParser().parse(["--log"])

        # Assert
        assert config.enable_logging is True

    def test_enables_streaming_with_the_stream_flag(self) -> None:
        # Act
        config = ArgumentParser().parse(["--stream"])

        # Assert
        assert config.enable_streaming is True

    def test_reads_the_provider_override(self) -> None:
        # Act
        config = ArgumentParser().parse(["--provider", "openai"])

        # Assert
        assert config.provider == "openai"

    def test_reads_overrides_from_argv(self) -> None:
        # Act
        config = ArgumentParser().parse(
            [
                "--prompt",
                "What is 2 + 2?",
                "--model-path",
                "/tmp/model.gguf",
                "--ctx",
                "4096",
                "--threads",
                "4",
                "--gpu-layers",
                "20",
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
            model_path=ModelPath(Path("/tmp/model.gguf")),
            context_size=ContextSize(4096),
            thread_count=ThreadCount(4),
            gpu_layer_count=GpuLayerCount(20),
            temperature=Temperature(0.7),
            max_tokens=MaxTokens(256),
            max_iterations=MaxIterations(3),
        )

    def test_rejects_invalid_argument_at_the_boundary(self) -> None:
        # Act / Assert: a negative thread count fails when the value object is built.
        with pytest.raises(ValueError, match="ThreadCount is not positive"):
            ArgumentParser().parse(["--threads", "0"])
