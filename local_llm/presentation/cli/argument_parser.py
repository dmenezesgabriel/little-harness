"""Parses command-line arguments into a validated `AppConfig`.

Building the config from value objects means invalid input (e.g. a negative
thread count) fails here, at the boundary, with a clear message.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

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

DEFAULT_MODEL_PATH = "models/LFM2-8B-A1B-Q4_K_M.gguf"
DEFAULT_PROVIDER = "llama_cpp"
DEFAULT_PROMPT = (
    "Explain llama.cpp in exactly 3 short bullet points. "
    "Be specific: mention GGUF models, local inference, "
    "and CPU-friendly execution."
)


class ArgumentParser:
    """Turns argv into an `AppConfig`.

    Example:
        config = ArgumentParser().parse(["--threads", "4"])
    """

    def parse(self, argv: Sequence[str] | None = None) -> AppConfig:
        namespace = build_parser().parse_args(argv)
        return to_app_config(namespace)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a small local LLM agent.")
    add_prompt_arguments(parser)
    add_model_arguments(parser)
    add_runtime_arguments(parser)
    return parser


def add_prompt_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-p",
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt to send to the local model.",
    )


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="Path to the local GGUF model.",
    )
    parser.add_argument("--ctx", type=int, default=8192, help="Context size.")
    parser.add_argument("--threads", type=int, default=8, help="CPU thread count.")
    parser.add_argument(
        "--gpu-layers",
        type=int,
        default=0,
        help="Number of GPU layers. Use 0 for CPU-only.",
    )


def add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Sampling temperature."
    )
    parser.add_argument(
        "--max-tokens", type=int, default=512, help="Maximum generated tokens."
    )
    parser.add_argument(
        "--max-iterations", type=int, default=5, help="Maximum agent loop iterations."
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Emit structured JSON logs for each agent event.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream generated tokens to stdout as they are produced.",
    )
    parser.add_argument(
        "--provider",
        default=DEFAULT_PROVIDER,
        help="Chat model provider to use.",
    )


def to_app_config(namespace: argparse.Namespace) -> AppConfig:
    return AppConfig(
        prompt=Prompt(str(namespace.prompt)),
        model_path=ModelPath(Path(str(namespace.model_path))),
        context_size=ContextSize(int(namespace.ctx)),
        thread_count=ThreadCount(int(namespace.threads)),
        gpu_layer_count=GpuLayerCount(int(namespace.gpu_layers)),
        temperature=Temperature(float(namespace.temperature)),
        max_tokens=MaxTokens(int(namespace.max_tokens)),
        max_iterations=MaxIterations(int(namespace.max_iterations)),
        provider=str(namespace.provider),
        enable_logging=bool(namespace.log),
        enable_streaming=bool(namespace.stream),
    )
