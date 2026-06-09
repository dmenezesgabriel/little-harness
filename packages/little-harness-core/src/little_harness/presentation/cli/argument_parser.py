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
    RepeatPenalty,
    Temperature,
    TopP,
)
from little_harness.domain.values.text_values import Prompt
from little_harness.presentation.cli.app_config import AppConfig

OPTION_SEPARATOR = "="
TOOL_SEPARATOR = ","


class ArgumentParser:
    """Turns argv into an `AppConfig`.

    Example:
        config = ArgumentParser().parse(["--provider", "litellm", "-o", "model=gpt"])

    """

    def parse(self, argv: Sequence[str] | None = None) -> AppConfig:
        """Parse command-line arguments into an ``AppConfig``."""
        namespace = build_parser().parse_args(argv)
        return to_app_config(namespace)


def build_parser() -> argparse.ArgumentParser:
    """Build and return the top-level argument parser."""
    parser = argparse.ArgumentParser(description="Run a small local LLM agent.")
    add_prompt_arguments(parser)
    add_runtime_arguments(parser)
    add_provider_arguments(parser)
    return parser


def add_prompt_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the --prompt / -p argument."""
    parser.add_argument(
        "-p",
        "--prompt",
        help="Prompt to send to the model.",
    )


def add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    """Register sampling, streaming, and loop-control arguments."""
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Sampling temperature."
    )
    parser.add_argument(
        "--top-p",
        type=float,
        help="Nucleus sampling threshold (0.0..1.0). Default lets provider decide.",
    )
    parser.add_argument(
        "--repeat-penalty",
        type=float,
        help="Repetition penalty (0.0..2.0, 1.0 = off). Provider default when unset.",
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
        help="Shorthand for --observer logging.",
    )
    parser.add_argument(
        "--observer",
        help="Observer plugin to use (an installed plugin name, e.g. 'logging'). "
        "Defaults to no observer.",
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
    parser.add_argument(
        "--ui",
        default="default",
        help="Interactive UI plugin to use (e.g. 'rich', 'default').",
    )


def add_provider_arguments(parser: argparse.ArgumentParser) -> None:
    """Register provider, policy, model, and provider-specific option arguments."""
    parser.add_argument(
        # No `default=`: argparse defaults to None, meaning "no provider chosen",
        # which the composition root resolves to the sole installed provider.
        "--provider",
        help="Chat model provider to use (an installed plugin name). "
        "Defaults to the sole installed provider.",
    )
    parser.add_argument(
        # No `default=`: None means "no policy chosen", which the composition root
        # resolves to the sole installed policy.
        "--policy",
        help="Agent policy plugin to use (an installed plugin name, e.g. 'json'). "
        "Defaults to the sole installed policy.",
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
    """Convert a parsed argparse namespace into an ``AppConfig``."""
    # argparse already applied each argument's `type=`/`action`; value objects then
    # validate ranges, and the provider validates its own options downstream.
    # prompt=None means interactive mode (REPL); a value means one-shot execution.
    prompt = Prompt(namespace.prompt) if namespace.prompt is not None else None
    return AppConfig(
        prompt=prompt,
        temperature=Temperature(namespace.temperature),
        max_tokens=MaxTokens(namespace.max_tokens),
        max_iterations=MaxIterations(namespace.max_iterations),
        top_p=TopP(namespace.top_p) if namespace.top_p is not None else None,
        repeat_penalty=RepeatPenalty(namespace.repeat_penalty)
        if namespace.repeat_penalty is not None
        else None,
        provider=namespace.provider,
        provider_options=build_provider_options(namespace.model, namespace.options),
        policy=namespace.policy,
        observer_name=resolve_observer_name(namespace.observer, namespace.log),
        enable_streaming=namespace.stream,
        tool_selection=parse_tool_selection(namespace.tools),
        approve_all=namespace.approve_all,
        ui=namespace.ui,
    )


def resolve_observer_name(observer: str | None, log: bool) -> str | None:
    """Return the observer plugin name from ``--observer`` or ``--log``."""
    # `--observer` is explicit; `--log` is the shorthand for the logging observer.
    if observer is not None:
        return observer

    return "logging" if log else None


def parse_tool_selection(raw: str | None) -> tuple[str, ...] | None:
    """Parse a comma-separated tool list into a tuple of names."""
    if raw is None:
        return None

    names = tuple(name.strip() for name in raw.split(TOOL_SEPARATOR))

    if any(name == "" for name in names):
        raise ValueError(
            f"Invalid --tools: {raw!r}. Expected comma-separated tool names."
        )

    return names


def build_provider_options(model: str | None, pairs: Sequence[str]) -> dict[str, str]:
    """Merge ``--model`` and ``--option`` pairs into a provider options dict."""
    # `--model` is shorthand for the `model` option; an explicit `-o model=` wins.
    options: dict[str, str] = {}
    if model is not None:
        options["model"] = model
    options.update(parse_options(pairs))
    return options


def parse_options(pairs: Sequence[str]) -> dict[str, str]:
    """Parse a sequence of KEY=VALUE strings into a dict."""
    return dict(split_option(pair) for pair in pairs)


def split_option(pair: str) -> tuple[str, str]:
    """Split a single KEY=VALUE string into a ``(key, value)`` tuple."""
    key, separator, value = pair.partition(OPTION_SEPARATOR)

    if separator == "" or key == "":
        raise ValueError(f"Invalid --option: {pair!r}. Expected KEY=VALUE.")

    return key, value
