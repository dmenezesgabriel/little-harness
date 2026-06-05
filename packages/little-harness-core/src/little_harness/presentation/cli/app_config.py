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
    Temperature,
)
from little_harness.domain.values.text_values import Prompt


def _no_options() -> dict[str, str]:
    return {}


@dataclass(frozen=True)
class AppConfig:
    prompt: Prompt
    temperature: Temperature
    max_tokens: MaxTokens
    max_iterations: MaxIterations
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
    # Skip interactive approval prompts and allow every sensitive tool to run.
    approve_all: bool = False
