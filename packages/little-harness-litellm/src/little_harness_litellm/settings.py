"""Construction settings for the LiteLLM provider, as a value object."""

from __future__ import annotations

from dataclasses import dataclass

from little_harness.domain.values.guards import require_non_empty_text


@dataclass(frozen=True)
class LiteLLMSettings:
    """Settings for a LiteLLM-backed chat model.

    `model` is a LiteLLM model string (e.g. "gpt-4o", "ollama/llama3"). The base
    URL and key are optional so hosted, proxied, and local backends all work.

    Example:
        settings = LiteLLMSettings("gpt-4o", api_base="https://proxy/v1")
    """

    model: str
    api_base: str | None = None
    api_key: str | None = None

    def __post_init__(self) -> None:
        normalized = require_non_empty_text(self.model, "LiteLLM model")
        object.__setattr__(self, "model", normalized)
