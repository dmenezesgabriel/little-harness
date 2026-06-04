"""Parses command-line arguments into a validated `AppConfig`.

The core CLI is provider-agnostic: sampling/loop flags are first-class, but every
provider-specific setting arrives through repeatable `--option KEY=VALUE` pairs and
is validated later by the selected provider plugin. Value objects reject invalid
ranges here, at the boundary, with a clear message.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from little_harness.domain.values.numeric_values import (
    MaxIterations,
    MaxTokens,
    Temperature,
)
from little_harness.domain.values.text_values import Prompt
from little_harness.presentation.cli.app_config import AppConfig

# Provider-agnostic default: exercises tool calling without naming any provider.
DEFAULT_PROMPT = "What is 144 divided by 12? Then tell me if the result is even or odd."
OPTION_SEPARATOR = "="
TOOL_SEPARATOR = ","


class ArgumentParser:
    """Turns argv into an `AppConfig`.

    Example:
        config = ArgumentParser().parse(["--provider", "litellm", "-o", "model=gpt"])
    """

    def parse(self, argv: Sequence[str] | None = None) -> AppConfig:
        namespace = build_parser().parse_args(argv)
        return to_app_config(namespace)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a small local LLM agent.")
    add_prompt_arguments(parser)
    add_runtime_arguments(parser)
    add_provider_arguments(parser)
    return parser


def add_prompt_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-p",
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt to send to the model.",
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
        "--tools",
        help="Comma-separated tool names to enable (installed plugin names). "
        "Defaults to every installed tool.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        dest="approve_all",
        help="Approve every sensitive tool without prompting (non-interactive runs).",
    )


def add_provider_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        # No `default=`: argparse defaults to None, meaning "no provider chosen",
        # which the composition root resolves to the sole installed provider.
        "--provider",
        help="Chat model provider to use (an installed plugin name). "
        "Defaults to the sole installed provider.",
    )
    parser.add_argument(
        "-m",
        "--model",
        help="Model to use; provider-specific (a model name for litellm, a GGUF "
        "path for llama_cpp). Shorthand for -o model=...",
    )
    parser.add_argument(
        "-o",
        "--option",
        action="append",
        default=[],
        dest="options",
        metavar="KEY=VALUE",
        help="Provider-specific setting, repeatable (e.g. -o n_ctx=8192).",
    )


def to_app_config(namespace: argparse.Namespace) -> AppConfig:
    # argparse already applied each argument's `type=`/`action`; value objects then
    # validate ranges, and the provider validates its own options downstream.
    return AppConfig(
        prompt=Prompt(namespace.prompt),
        temperature=Temperature(namespace.temperature),
        max_tokens=MaxTokens(namespace.max_tokens),
        max_iterations=MaxIterations(namespace.max_iterations),
        provider=namespace.provider,
        provider_options=build_provider_options(namespace.model, namespace.options),
        enable_logging=namespace.log,
        enable_streaming=namespace.stream,
        tool_selection=parse_tool_selection(namespace.tools),
        approve_all=namespace.approve_all,
    )


def parse_tool_selection(raw: str | None) -> tuple[str, ...] | None:
    if raw is None:
        return None

    names = tuple(name.strip() for name in raw.split(TOOL_SEPARATOR))

    if any(name == "" for name in names):
        raise ValueError(
            f"Invalid --tools: {raw!r}. Expected comma-separated tool names."
        )

    return names


def build_provider_options(model: str | None, pairs: Sequence[str]) -> dict[str, str]:
    # `--model` is shorthand for the `model` option; an explicit `-o model=` wins.
    options: dict[str, str] = {}
    if model is not None:
        options["model"] = model
    options.update(parse_options(pairs))
    return options


def parse_options(pairs: Sequence[str]) -> dict[str, str]:
    return dict(split_option(pair) for pair in pairs)


def split_option(pair: str) -> tuple[str, str]:
    key, separator, value = pair.partition(OPTION_SEPARATOR)

    if separator == "" or key == "":
        raise ValueError(f"Invalid --option: {pair!r}. Expected KEY=VALUE.")

    return key, value
