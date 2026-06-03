"""Adapter implementing the `ChatModel` port over a llama.cpp model."""

from __future__ import annotations

from typing import cast

from llama_cpp.llama_types import CreateChatCompletionResponse

from local_llm.application.ports.chat_model import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from local_llm.domain.values.text_values import MessageContent
from local_llm.infrastructure.llama_cpp.message_mapper import to_llama_message
from local_llm.infrastructure.llama_cpp.model_factory import create_llama_model
from local_llm.infrastructure.llama_cpp.settings import LlamaCppModelSettings


class LlamaCppChatModel:
    """Runs chat completions on a local GGUF model via llama.cpp.

    Example:
        response = LlamaCppChatModel(settings).complete(request)
    """

    def __init__(self, settings: LlamaCppModelSettings) -> None:
        self._llm = create_llama_model(settings)

    def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        response = self._llm.create_chat_completion(
            messages=[to_llama_message(message) for message in request.messages],
            temperature=request.temperature.value,
            max_tokens=request.max_tokens.value,
        )
        content = extract_response_content(
            cast("CreateChatCompletionResponse", response)
        )
        return ChatCompletionResponse(MessageContent(content))


def extract_response_content(response: CreateChatCompletionResponse) -> str:
    choices = response["choices"]

    if len(choices) == 0:
        raise ValueError("Expected at least one response choice, got empty list.")

    content = choices[0]["message"]["content"]

    if not isinstance(content, str):
        raise TypeError(f"Expected message content string, got: {type(content)}")

    return content.strip()
