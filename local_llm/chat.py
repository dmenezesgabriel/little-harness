from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str


@dataclass(frozen=True)
class ChatCompletionRequest:
    messages: Sequence[ChatMessage]
    temperature: float
    max_tokens: int


@dataclass(frozen=True)
class ChatCompletionResponse:
    content: str


class ChatModel(Protocol):
    def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Return a chat completion for the given messages.

        Example:
            response = model.complete(ChatCompletionRequest(messages, 0.0, 512))
        """
        ...
