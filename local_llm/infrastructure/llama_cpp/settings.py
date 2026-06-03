"""Construction settings for the llama.cpp model, as value objects."""

from __future__ import annotations

from dataclasses import dataclass

from local_llm.domain.values.model_path import ModelPath
from local_llm.domain.values.numeric_values import (
    ContextSize,
    GpuLayerCount,
    ThreadCount,
)


@dataclass(frozen=True)
class LlamaCppModelSettings:
    model_path: ModelPath
    context_size: ContextSize
    thread_count: ThreadCount
    gpu_layer_count: GpuLayerCount
