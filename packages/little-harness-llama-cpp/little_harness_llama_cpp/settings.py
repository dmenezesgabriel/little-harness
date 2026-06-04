"""Construction settings for the llama.cpp model, as value objects."""

from __future__ import annotations

from dataclasses import dataclass

from little_harness_llama_cpp.values import (
    ContextSize,
    GpuLayerCount,
    ModelPath,
    ThreadCount,
)


@dataclass(frozen=True)
class LlamaCppModelSettings:
    model_path: ModelPath
    context_size: ContextSize
    thread_count: ThreadCount
    gpu_layer_count: GpuLayerCount
