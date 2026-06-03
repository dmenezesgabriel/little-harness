"""Named test doubles for the llama.cpp adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from llama_cpp.llama_types import CreateChatCompletionResponse

from local_llm.domain.values.model_path import ModelPath
from local_llm.domain.values.numeric_values import (
    ContextSize,
    GpuLayerCount,
    ThreadCount,
)
from local_llm.infrastructure.llama_cpp.settings import LlamaCppModelSettings


class FakeLlama:
    """Stand-in for llama_cpp.Llama that records constructor and call arguments."""

    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self.completion_kwargs: dict[str, Any] = {}

    def create_chat_completion(
        self,
        *,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
    ) -> CreateChatCompletionResponse:
        self.completion_kwargs = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        return cast(
            "CreateChatCompletionResponse",
            {"choices": [{"message": {"content": " hi there "}}]},
        )


def make_settings(model_path: Path) -> LlamaCppModelSettings:
    return LlamaCppModelSettings(
        model_path=ModelPath(model_path),
        context_size=ContextSize(8192),
        thread_count=ThreadCount(8),
        gpu_layer_count=GpuLayerCount(0),
    )
