"""Named test doubles for the llama.cpp adapter."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

from little_harness_llama_cpp.settings import LlamaCppModelSettings
from little_harness_llama_cpp.values import (
    ContextSize,
    GpuLayerCount,
    ModelPath,
    ThreadCount,
)
from llama_cpp.llama_types import CreateChatCompletionStreamResponse


class FakeLlama:
    """Stand-in for llama_cpp.Llama that records constructor and call arguments."""

    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self.completion_kwargs: dict[str, Any] = {}
        self.closed = False

    def close(self) -> None:
        self.closed = True

    def create_chat_completion(
        self,
        *,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
        stream: bool = False,
    ) -> Iterator[CreateChatCompletionStreamResponse]:
        self.completion_kwargs = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        # First chunk is role-only (no content); the adapter must skip it.
        return iter(
            [
                cast(
                    "CreateChatCompletionStreamResponse",
                    {"choices": [{"delta": {"role": "assistant"}}]},
                ),
                cast(
                    "CreateChatCompletionStreamResponse",
                    {"choices": [{"delta": {"content": " hi there "}}]},
                ),
            ]
        )


class NonStreamingLlama:
    """Stand-in whose create_chat_completion returns a non-iterator response."""

    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self.closed = False

    def close(self) -> None:
        self.closed = True

    def create_chat_completion(self, **_kwargs: Any) -> dict[str, Any]:
        return {"choices": [{"message": {"content": "whole answer"}}]}


def make_settings(model_path: Path) -> LlamaCppModelSettings:
    return LlamaCppModelSettings(
        model_path=ModelPath(model_path),
        context_size=ContextSize(8192),
        thread_count=ThreadCount(8),
        gpu_layer_count=GpuLayerCount(0),
    )
