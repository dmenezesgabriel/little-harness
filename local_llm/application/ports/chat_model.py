"""Port for a chat-completion model, plus its request/response DTOs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from local_llm.domain.message_history import MessageHistory
from local_llm.domain.values.numeric_values import MaxTokens, Temperature
from local_llm.domain.values.text_values import MessageContent


@dataclass(frozen=True)
class ChatCompletionRequest:
    messages: MessageHistory
    temperature: Temperature
    max_tokens: MaxTokens


@dataclass(frozen=True)
class ChatCompletionResponse:
    content: MessageContent


class ChatModel(Protocol):
    def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Return a chat completion for the given conversation.

        Example:
            response = model.complete(request)
        """
        ...
