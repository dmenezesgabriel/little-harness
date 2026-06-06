"""Port for a chat-completion model, plus its request DTO."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Protocol

from little_harness.application.ports.closeable import Closeable
from little_harness.domain.message_history import MessageHistory
from little_harness.domain.values.numeric_values import (
    MaxTokens,
    RepeatPenalty,
    Temperature,
    TopP,
)
from little_harness.domain.values.text_values import MessageContent


@dataclass(frozen=True)
class ResponseSchema:
    """A JSON Schema that constrains model output to one parseable shape.

    The policy owns the protocol, so it produces this; providers that support
    constrained decoding (llama.cpp grammars, OpenAI json_schema) use it to make
    invalid output structurally impossible, while others ignore it and lean on
    the prompt plus a lenient parser.

    Example:
        schema = ResponseSchema({"type": "object", "required": ["action"]})
    """

    value: Mapping[str, object]


@dataclass(frozen=True)
class ChatCompletionRequest:
    messages: MessageHistory
    temperature: Temperature
    max_tokens: MaxTokens
    response_schema: ResponseSchema | None = None
    top_p: TopP | None = None
    repeat_penalty: RepeatPenalty | None = None


class ChatModel(Closeable, Protocol):
    """A chat-completion model. Owns native resources, so it is `Closeable`."""

    def complete_streaming(
        self, request: ChatCompletionRequest
    ) -> Iterator[MessageContent]:
        """Stream the completion as content chunks as they are generated.

        The caller joins the chunks for the full text and may forward each to a
        `TokenSink` for live output. The same method serves single-prompt and
        interactive modes.

        Example:
            text = "".join(chunk.value for chunk in model.complete_streaming(req))
        """
        ...
