"""Entry-point builder: turn provider options into a LiteLLM `ChatModel`.

Registered under the `little_harness.chat_model_providers` group as `litellm`.

Example:
    model = build({"model": "gpt-4o", "api_base": "https://proxy/v1"})

"""

from __future__ import annotations

from collections.abc import Mapping

from little_harness.application.ports.chat_model import ChatModel

from little_harness_litellm.chat_model import LiteLLMChatModel
from little_harness_litellm.settings import LiteLLMSettings


def build(options: Mapping[str, str]) -> ChatModel:
    """Build a LiteLLM-backed ChatModel from provider options."""
    return LiteLLMChatModel(to_settings(options))


def to_settings(options: Mapping[str, str]) -> LiteLLMSettings:
    """Parse provider options into a validated LiteLLMSettings, requiring a model."""
    model = options.get("model")

    if model is None:
        raise ValueError(
            "Option 'model' is required for the litellm provider. "
            "Expected e.g. -o model=gpt-4o."
        )

    return LiteLLMSettings(
        model=model,
        api_base=options.get("api_base"),
        api_key=options.get("api_key"),
        num_retries=int_option(options, "num_retries", 0),
    )


def int_option(options: Mapping[str, str], key: str, default: int) -> int:
    """Extract an integer option from a string-keyed mapping, or return the default."""
    raw = options.get(key)

    if raw is None:
        return default

    try:
        return int(raw)
    except ValueError as error:
        raise ValueError(
            f"Option {key!r} is not an integer: {raw!r}. Expected a base-10 integer."
        ) from error
