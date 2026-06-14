"""Raw-config dataclass consumed by ArgumentParser and produced by ConfigLoader."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


def _no_plugins() -> dict[str, dict[str, str]]:
    return {}


@dataclass(frozen=True)
class Config:
    """Layered configuration merged from code defaults, TOML, profiles, and CLI args."""

    temperature: float | None = None
    max_tokens: int | None = None
    max_iterations: int | None = None
    top_p: float | None = None
    repeat_penalty: float | None = None
    provider: str | None = None
    model: str | None = None
    policy: str | None = None
    observer: str | None = None
    stream: bool | None = None
    tools: tuple[str, ...] | None = None
    approve_all: bool | None = None
    ui: str | None = None
    profile: str | None = None
    plugins: Mapping[str, Mapping[str, str]] = field(default_factory=_no_plugins)
