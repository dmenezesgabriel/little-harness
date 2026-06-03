"""Parsed CLI configuration, expressed entirely in value objects."""

from __future__ import annotations

from dataclasses import dataclass

from local_llm.domain.values.model_path import ModelPath
from local_llm.domain.values.numeric_values import (
    ContextSize,
    GpuLayerCount,
    MaxIterations,
    MaxTokens,
    Temperature,
    ThreadCount,
)
from local_llm.domain.values.text_values import Prompt


@dataclass(frozen=True)
class AppConfig:
    prompt: Prompt
    model_path: ModelPath
    context_size: ContextSize
    thread_count: ThreadCount
    gpu_layer_count: GpuLayerCount
    temperature: Temperature
    max_tokens: MaxTokens
    max_iterations: MaxIterations
