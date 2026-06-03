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
from local_llm.presentation.cli.argument_parser import ArgumentParser


class TestArgumentParser:
    def test_uses_defaults_when_no_flags_given(self) -> None:
        # Act
        config = ArgumentParser().parse([])

        # Assert: prompt compared separately because the default is a long string.
        assert config.model_path == ModelPath(Path("models/LFM2-8B-A1B-Q4_K_M.gguf"))
        assert config.context_size == ContextSize(8192)
        assert config.thread_count == ThreadCount(8)
        assert config.gpu_layer_count == GpuLayerCount(0)
        assert config.temperature == Temperature(0.0)
        assert config.max_tokens == MaxTokens(512)
        assert config.max_iterations == MaxIterations(5)
        assert "llama.cpp" in config.prompt.value

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
