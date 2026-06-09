"""Adapter implementing the `ChatModel` port over a llama.cpp model."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from little_harness.application.ports.chat_model import (
    ChatCompletionRequest,
    ResponseSchema,
)
from little_harness.domain.values.text_values import MessageContent
from llama_cpp.llama_types import (
    ChatCompletionRequestResponseFormat,
    CreateChatCompletionStreamResponse,
)

from little_harness_llama_cpp.message_mapper import to_llama_message
from little_harness_llama_cpp.model_factory import create_llama_model
from little_harness_llama_cpp.settings import LlamaCppModelSettings


class LlamaCppChatModel:
    """Streams chat completions from a local GGUF model via llama.cpp.

    Example:
        chunks = LlamaCppChatModel(settings).complete_streaming(request)

    """

    def __init__(self, settings: LlamaCppModelSettings) -> None:
        """See class docstring for argument descriptions."""
        self._llm = create_llama_model(settings)

    def complete_streaming(
        self, request: ChatCompletionRequest
    ) -> Iterator[MessageContent]:
        """Stream completion chunks for the given request."""
        completion_kwargs: dict[str, Any] = {
            "messages": [to_llama_message(message) for message in request.messages],
            "temperature": request.temperature.value,
            "max_tokens": request.max_tokens.value,
            "stream": True,
            "response_format": to_response_format(request.response_schema),
        }
        if request.top_p is not None:
            completion_kwargs["top_p"] = request.top_p.value
        if request.repeat_penalty is not None:
            completion_kwargs["repeat_penalty"] = request.repeat_penalty.value
        stream = self._llm.create_chat_completion(**completion_kwargs)
        # `stream=True` returns an iterator, but the SDK's return type is a union
        # with the non-streaming response; narrow it so chunks are typed.
        if not isinstance(stream, Iterator):
            raise TypeError(f"Expected a streaming response, got: {type(stream)}")

        for chunk in stream:
            content = extract_chunk_content(chunk)
            if content is not None:
                yield MessageContent(content)

    def close(self) -> None:
        """Close the underlying model."""
        self._llm.close()


def to_response_format(
    schema: ResponseSchema | None,
) -> ChatCompletionRequestResponseFormat | None:
    """Turn a policy schema into llama.cpp's JSON-grammar response format.

    Passing `schema` makes llama.cpp build a GBNF grammar that constrains decoding
    to schema-valid JSON, so malformed output (and its repair round-trips) cannot
    happen. None leaves generation unconstrained.
    """
    if schema is None:
        return None

    return ChatCompletionRequestResponseFormat(
        type="json_object", schema=dict(schema.value)
    )


def extract_chunk_content(chunk: CreateChatCompletionStreamResponse) -> str | None:
    """Extract text content from a streaming chunk, or return None."""
    choices = chunk["choices"]

    if len(choices) == 0:
        return None

    content = choices[0]["delta"].get("content")

    if content is None:
        return None

    if not isinstance(content, str):
        raise TypeError(f"Expected streamed content string, got: {type(content)}")

    return content
