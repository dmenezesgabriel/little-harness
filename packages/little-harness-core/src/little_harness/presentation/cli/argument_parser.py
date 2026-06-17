"""Parses command-line arguments into a validated `AppConfig`.

The core CLI is provider-agnostic: sampling/loop flags are first-class, but every
provider-specific setting arrives through repeatable `--option KEY=VALUE` pairs and
is validated later by the selected provider plugin. Value objects reject invalid
ranges here, at the boundary, with a clear message.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Any

from little_harness.infrastructure.config.config_types import Config
from little_harness.domain.values.numeric_values import (
    MaxIterations,
    MaxTokens,
    RepeatPenalty,
    Temperature,
    TopP,
)
from little_harness.domain.values.text_values import Prompt, SessionId
from little_harness.presentation.cli.app_config import AppConfig

OPTION_SEPARATOR = "="
TOOL_SEPARATOR = ","

# Code defaults for every CLI-settable field.
# These are the floor: config.toml overrides them, CLI args override config.
_CODE_DEFAULTS: dict[str, Any] = {
    "temperature": 0.1,
    "max_tokens": 512,
    "max_iterations": 5,
    "stream": False,
    "approve_all": False,
    "ui": "default",
    "log": False,
}

# Mapping from Config field names to the values-dict keys used in the merge.
# Most Config fields map directly; a few are aliased.
_CONFIG_FIELD_MAP: dict[str, str] = {
    "temperature": "temperature",
    "max_tokens": "max_tokens",
    "max_iterations": "max_iterations",
    "top_p": "top_p",
    "repeat_penalty": "repeat_penalty",
    "provider": "provider",
    "model": "model",
    "policy": "policy",
    "observer": "observer",
    "stream": "stream",
    "tools": "tools",
    "approve_all": "approve_all",
    "ui": "ui",
    "session_id": "session_id",
}

# CLI dests that have special handling not covered by the generic loop.
_CLI_SKIP_KEYS = frozenset({"options", "model"})

# `store_true` args: the CLI value is False when the flag is absent and True
# when present. Only apply the True case so a missing flag does not overwrite
# a config-file value of true.
_BOOLEAN_FLAGS = frozenset({"stream", "log", "approve_all"})


class ArgumentParser:
    """Turns argv into an `AppConfig`, optionally overlaying a ``Config``.

    Example:
        config = ArgumentParser().parse(["--provider", "litellm", "-o", "model=gpt"])

    """

    def __init__(self, config: Config | None = None) -> None:
        """Store an optional Config whose non-None fields act as defaults."""
        self._config = config

    def parse(self, argv: Sequence[str] | None = None) -> AppConfig:
        """Parse command-line arguments into an ``AppConfig``."""
        namespace = build_parser().parse_args(argv)
        return to_app_config(namespace, self._config)


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
    """Register sampling, streaming, loop-control, and profile arguments."""
    # All `store` args default to None so _merge_cli can distinguish
    # "user explicitly passed" from "default was used". Code defaults
    # live in _CODE_DEFAULTS and are applied by _merge_config.
    parser.add_argument(
        "--profile",
        help="Profile to activate (overrides config default).",
    )
    parser.add_argument(
        "--temperature", type=float, help="Sampling temperature. Default: 0.1."
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
        "--max-tokens", type=int, help="Maximum generated tokens. Default: 512."
    )
    parser.add_argument(
        "--max-iterations", type=int, help="Maximum agent loop iterations. Default: 5."
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
        help="Interactive UI plugin to use (e.g. 'rich', 'default'). Default: default.",
    )
    parser.add_argument(
        "-s",
        "--session",
        dest="session_id",
        help="Session ID to resume (a previously saved session identifier).",
    )


def add_provider_arguments(parser: argparse.ArgumentParser) -> None:
    """Register provider, policy, model, and provider-specific option arguments."""
    parser.add_argument(
        "--provider",
        help="Chat model provider to use (an installed plugin name). "
        "Defaults to the sole installed provider.",
    )
    parser.add_argument(
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


def to_app_config(
    namespace: argparse.Namespace, config: Config | None = None
) -> AppConfig:
    """Convert a parsed argparse namespace into an ``AppConfig``.

    Values resolve in this order (each overrides the previous):
    1. | ``_CODE_DEFAULTS`` — built-in floor
    2. ``Config`` — TOML file values (only non-None fields)
    3. CLI ``namespace`` — explicit user flags
    """
    merged = _merge_config(config)
    _merge_cli(merged, namespace)

    prompt = Prompt(merged["prompt"]) if merged.get("prompt") is not None else None
    return AppConfig(
        prompt=prompt,
        temperature=Temperature(merged["temperature"]),
        max_tokens=MaxTokens(merged["max_tokens"]),
        max_iterations=MaxIterations(merged["max_iterations"]),
        top_p=TopP(merged["top_p"]) if merged.get("top_p") is not None else None,
        repeat_penalty=RepeatPenalty(merged["repeat_penalty"])
        if merged.get("repeat_penalty") is not None
        else None,
        provider=merged.get("provider"),
        provider_options=build_provider_options(
            merged.get("model"), merged.get("options", [])
        ),
        policy=merged.get("policy"),
        observer_name=resolve_observer_name(
            merged.get("observer"), merged.get("log", False)
        ),
        enable_streaming=merged.get("stream", False),
        tool_selection=parse_tool_selection(merged.get("tools")),
        approve_all=merged.get("approve_all", False),
        ui=merged.get("ui", "default"),
        profile=merged.get("profile"),
        session_id=SessionId(merged["session_id"])
        if merged.get("session_id") is not None
        else None,
        plugin_configs=config.plugins if config is not None else {},
        skill_paths=merged.get("skill_paths", (".agents/skills",)),
    )


def _merge_config(config: Config | None) -> dict[str, Any]:
    """Apply code defaults, then overlay Config values."""
    merged: dict[str, Any] = dict(_CODE_DEFAULTS)

    if config is None:
        return merged

    for config_key, values_key in _CONFIG_FIELD_MAP.items():
        value = getattr(config, config_key, None)
        if value is not None:
            merged[values_key] = value

    return merged


def _merge_cli(merged: dict[str, Any], namespace: argparse.Namespace) -> None:
    """Overlay CLI values on top of the merged dict.

    * ``store`` args: only applied when the value is not None (i.e. user
      explicitly passed the flag).
    * ``store_true`` args: only applied when the value is True (False is
      the argparse default and means the flag was absent).
    * ``model`` / ``options``: handled after the loop via special merge.
    """
    for key, cli_value in vars(namespace).items():
        if key in _CLI_SKIP_KEYS:
            continue
        if key in _BOOLEAN_FLAGS:
            if cli_value is True:
                merged[key] = True
            continue
        if cli_value is not None:
            merged[key] = cli_value

    # model and options get special merge: CLI model overrides config model,
    # CLI -o KEY=VALUE pairs are accumulated for provider_options.
    model = getattr(namespace, "model", None)
    if model is not None:
        merged["model"] = model

    options = getattr(namespace, "options", None)
    if options:
        merged.setdefault("options", [])
        merged["options"].extend(options)


def resolve_observer_name(observer: str | None, log: bool) -> str | None:
    """Return the observer plugin name from ``--observer`` or ``--log``."""
    if observer is not None:
        return observer

    return "logging" if log else None


def parse_tool_selection(raw: str | tuple[str, ...] | None) -> tuple[str, ...] | None:
    """Parse a comma-separated tool list or tuple into a tuple of names."""
    if raw is None:
        return None

    if isinstance(raw, tuple):
        return raw

    names = tuple(name.strip() for name in raw.split(TOOL_SEPARATOR))

    if any(name == "" for name in names):
        raise ValueError(
            f"Invalid --tools: {raw!r}. Expected comma-separated tool names."
        )

    return names


def build_provider_options(model: str | None, pairs: Sequence[str]) -> dict[str, str]:
    """Merge ``--model`` and ``--option`` pairs into a provider options dict."""
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
