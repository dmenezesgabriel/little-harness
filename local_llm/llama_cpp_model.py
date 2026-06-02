from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

from llama_cpp import Llama
from llama_cpp.llama_types import (
    ChatCompletionRequestAssistantMessage,
    ChatCompletionRequestMessage,
    ChatCompletionRequestSystemMessage,
    ChatCompletionRequestUserMessage,
    CreateChatCompletionResponse,
)

from local_llm.chat import ChatCompletionRequest, ChatCompletionResponse, ChatMessage


@dataclass(frozen=True)
class LlamaCppModelSettings:
    model_path: Path
    context_size: int
    thread_count: int
    gpu_layer_count: int


class LlamaCppChatModel:
    def __init__(self, settings: LlamaCppModelSettings) -> None:
        self._llm = create_llama_model(settings)

    def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        response = self._llm.create_chat_completion(
            messages=[to_llama_message(message) for message in request.messages],
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        return ChatCompletionResponse(
            content=extract_response_content(
                cast("CreateChatCompletionResponse", response)
            )
        )


def create_llama_model(settings: LlamaCppModelSettings) -> Llama:
    if not settings.model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {settings.model_path}. "
            "Expected a local GGUF model file."
        )

    return Llama(
        model_path=str(settings.model_path),
        n_ctx=settings.context_size,
        n_threads=settings.thread_count,
        n_gpu_layers=settings.gpu_layer_count,
        verbose=False,
    )


def to_llama_message(message: ChatMessage) -> ChatCompletionRequestMessage:
    if message.role == "system":
        return ChatCompletionRequestSystemMessage(
            role="system",
            content=message.content,
        )

    if message.role == "assistant":
        return ChatCompletionRequestAssistantMessage(
            role="assistant",
            content=message.content,
        )

    return ChatCompletionRequestUserMessage(role="user", content=message.content)


def extract_response_content(response: CreateChatCompletionResponse) -> str:
    choices = response["choices"]
    if len(choices) == 0:
        raise ValueError("Expected at least one response choice, got empty list.")

    content = choices[0]["message"]["content"]

    if not isinstance(content, str):
        raise TypeError(f"Expected message content string, got: {type(content)}")

    return content.strip()
