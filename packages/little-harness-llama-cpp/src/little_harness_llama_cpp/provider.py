"""Entry-point builder: turn provider options into a ready `ChatModel`.

Registered under the `little_harness.chat_model_providers` group as `llama_cpp`.
The core composition root calls `build(options)`; this module reads and validates
the llama.cpp-specific option keys and constructs the adapter.

Example:
    model = build({"model_path": "models/m.gguf", "n_ctx": "8192"})
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from little_harness.application.ports.chat_model import ChatModel

from little_harness_llama_cpp.chat_model import LlamaCppChatModel
from little_harness_llama_cpp.settings import LlamaCppModelSettings
from little_harness_llama_cpp.values import (
    BatchSize,
    ContextSize,
    GpuLayerCount,
    ModelPath,
    ThreadCount,
)

DEFAULT_MODEL_PATH = "models/LFM2-8B-A1B-Q4_K_M.gguf"
DEFAULT_CONTEXT_SIZE = 8192
DEFAULT_THREAD_COUNT = 8
DEFAULT_GPU_LAYER_COUNT = 0
DEFAULT_BATCH_SIZE = 512
DEFAULT_FLASH_ATTENTION = True
TRUE_OPTION_VALUES = frozenset({"1", "true", "yes", "on"})


def build(options: Mapping[str, str]) -> ChatModel:
    return LlamaCppChatModel(to_settings(options))


def to_settings(options: Mapping[str, str]) -> LlamaCppModelSettings:
    # The GGUF path is the model here, so accept the generic `model` key (set by
    # --model) as an alias for the explicit `model_path` option.
    model_path = options.get("model_path") or options.get("model") or DEFAULT_MODEL_PATH
    return LlamaCppModelSettings(
        model_path=ModelPath(Path(model_path)),
        context_size=ContextSize(int_option(options, "n_ctx", DEFAULT_CONTEXT_SIZE)),
        thread_count=ThreadCount(
            int_option(options, "n_threads", DEFAULT_THREAD_COUNT)
        ),
        gpu_layer_count=GpuLayerCount(
            int_option(options, "n_gpu_layers", DEFAULT_GPU_LAYER_COUNT)
        ),
        batch_size=BatchSize(int_option(options, "n_batch", DEFAULT_BATCH_SIZE)),
        flash_attention=bool_option(options, "flash_attn", DEFAULT_FLASH_ATTENTION),
    )


def bool_option(options: Mapping[str, str], key: str, default: bool) -> bool:
    raw = options.get(key)

    if raw is None:
        return default

    return raw.strip().lower() in TRUE_OPTION_VALUES


def int_option(options: Mapping[str, str], key: str, default: int) -> int:
    raw = options.get(key)

    if raw is None:
        return default

    try:
        return int(raw)
    except ValueError as error:
        raise ValueError(
            f"Option {key!r} is not an integer: {raw!r}. Expected a base-10 integer."
        ) from error
