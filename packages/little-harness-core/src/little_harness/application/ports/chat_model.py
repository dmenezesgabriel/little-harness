"""Port for a chat-completion model, plus its request DTO."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

from little_harness.application.ports.closeable import Closeable
from little_harness.domain.message_history import MessageHistory
from little_harness.domain.values.numeric_values import MaxTokens, Temperature
from little_harness.domain.values.text_values import MessageContent


@dataclass(frozen=True)
class ChatCompletionRequest:
    messages: MessageHistory
    temperature: Temperature
    max_tokens: MaxTokens


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
