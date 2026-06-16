"""Parsed CLI configuration, provider-agnostic.

Provider-specific settings (model path, context size, api keys, ...) are carried
opaquely in `provider_options`; each provider plugin reads and validates its own
keys. The core never knows any provider's settings shape.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from little_harness.domain.values.numeric_values import (
    MaxIterations,
    MaxTokens,
    RepeatPenalty,
    Temperature,
    TopP,
)
from little_harness.domain.values.text_values import Prompt, SessionId


def _no_options() -> dict[str, str]:
    return {}


def _empty_plugin_configs() -> dict[str, dict[str, str]]:
    return {}


@dataclass(frozen=True)
class AppConfig:
    """Complete CLI configuration after argument parsing.

    Provider-specific values are carried opaquely in ``provider_options``;
    each provider plugin validates its own keys.
    """

    temperature: Temperature
    max_tokens: MaxTokens
    max_iterations: MaxIterations
    # None means interactive mode (REPL); a Prompt means one-shot execution.
    prompt: Prompt | None = None
    # None means "no provider chosen"; the composition root resolves the default.
    provider: str | None = None
    provider_options: Mapping[str, str] = field(default_factory=_no_options)
    # None means "no policy chosen"; the composition root resolves the default.
    policy: str | None = None
    # None means "no observer"; a name selects an installed observer plugin.
    observer_name: str | None = None
    enable_streaming: bool = False
    # None means "every installed tool"; a tuple limits discovery to those names.
    tool_selection: tuple[str, ...] | None = None
    # None means the provider uses its own default.
    top_p: TopP | None = None
    repeat_penalty: RepeatPenalty | None = None
    # Skip interactive approval prompts and allow every sensitive tool to run.
    approve_all: bool = False
    # Interactive UI plugin to use (e.g. 'rich', 'default').
    ui: str = "default"
    # Active profile name (None means no profile applied).
    profile: str | None = None
    # Plugin-specific config from TOML [plugins.*] sections.
    plugin_configs: Mapping[str, Mapping[str, str]] = field(
        default_factory=_empty_plugin_configs
    )
    # Session ID for resuming a previous session.
    session_id: SessionId | None = None
    # Directories containing SKILL.md skill files, relative to the project root.
    skill_paths: tuple[str, ...] = (".agents/skills",)
