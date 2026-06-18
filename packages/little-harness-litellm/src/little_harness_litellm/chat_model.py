"""Adapter implementing the `ChatModel` port over LiteLLM.

LiteLLM is a thin, stateless client over many providers, so `close()` is a no-op.
The vendor import is kept inside this package; the rest of the harness sees only
the `ChatModel` port.

Example:
    chunks = LiteLLMChatModel(settings).complete_streaming(request)

"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import litellm
from little_harness.application.ports.chat_model import (
    ChatCompletionRequest,
    ResponseSchema,
)
from little_harness.domain.values.text_values import MessageContent

from little_harness_litellm.message_mapper import to_litellm_message
from little_harness_litellm.settings import LiteLLMSettings

# Vendor boundary: LiteLLM ships partial type information, so route calls through an
# explicit `Any` view of the module (the module object itself is fully typed, so this
# assignment is clean). The streaming result is validated at runtime below — the one
# dynamic-typing edge in this adapter, mirroring the entry-point loader in core.
_litellm: Any = litellm


class LiteLLMChatModel:
    """Streams chat completions from any LiteLLM-supported backend.

    Example:
        model = LiteLLMChatModel(LiteLLMSettings("gpt-4o"))

    """

    def __init__(self, settings: LiteLLMSettings) -> None:
        """See class docstring for argument descriptions."""
        self._settings = settings

    def complete_streaming(
        self, request: ChatCompletionRequest
    ) -> Iterator[MessageContent]:
        """Stream completion chunks from the configured LiteLLM backend."""
        completion_kwargs: dict[str, Any] = {
            "model": self._settings.model,
            "messages": [to_litellm_message(message) for message in request.messages],
            "temperature": request.temperature.value,
            "max_tokens": request.max_tokens.value,
            "stream": True,
            "api_base": self._settings.api_base,
            "api_key": self._settings.api_key,
            "num_retries": self._settings.num_retries,
            "response_format": to_response_format(request.response_schema),
        }
        if request.top_p is not None:
            completion_kwargs["top_p"] = request.top_p.value
        if request.repeat_penalty is not None:
            completion_kwargs["repeat_penalty"] = request.repeat_penalty.value
        response = _litellm.completion(**completion_kwargs)
        # `stream=True` yields an iterator, but the SDK's return type unions it with
        # the non-streaming response; reject the non-streaming case at runtime. The
        # check lives in a helper so narrowing stays there and `response` remains `Any`.
        reject_non_streaming(response)

        for chunk in response:
            content = extract_delta_content(chunk)
            if content is not None:
                yield MessageContent(content)

    def supports_thinking(self) -> bool:
        """Return True for Anthropic models that support extended thinking."""
        model = self._settings.model.lower()
        return "anthropic" in model or "claude" in model

    def close(self) -> None:
        """LiteLLM holds no persistent native resource; nothing to release."""


def to_response_format(
    schema: ResponseSchema | None,
) -> dict[str, object] | None:
    """Turn a policy schema into LiteLLM's OpenAI-style `json_schema` format.

    Backends with structured outputs or guided decoding (OpenAI, vLLM, Ollama)
    enforce it; LiteLLM degrades it for the rest. None leaves output free.
    """
    if schema is None:
        return None

    return {
        "type": "json_schema",
        "json_schema": {"name": "agent_decision", "schema": dict(schema.value)},
    }


def reject_non_streaming(response: object) -> None:
    """Raise if the LiteLLM response is not an iterator (streaming was requested)."""
    if not isinstance(response, Iterator):
        raise TypeError(f"Expected a streaming response, got: {type(response)}")


def extract_delta_content(chunk: object) -> str | None:
    """Extract the text delta from a LiteLLM streaming chunk, or None if finished."""
    choices = getattr(chunk, "choices", None)

    if not choices:
        return None

    content = getattr(getattr(choices[0], "delta", None), "content", None)

    if content is None:
        return None

    if not isinstance(content, str):
        raise TypeError(f"Expected streamed content string, got: {type(content)}")

    return content
