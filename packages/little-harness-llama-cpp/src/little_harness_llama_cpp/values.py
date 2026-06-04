"""Value objects for llama.cpp construction settings.

These live in the provider plugin, not core: context window, thread count, GPU
offload, and the GGUF model path are llama.cpp concerns. They reuse the core
domain's shared guards so the exception shape stays consistent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from little_harness.domain.values.guards import (
    require_non_negative_int,
    require_positive_int,
)

GGUF_SUFFIX = ".gguf"


@dataclass(frozen=True)
class ModelPath:
    """Path to a local GGUF model. Existence is checked at load time, not here.

    Example:
        model_path = ModelPath(Path("models/model.gguf"))
    """

    value: Path

    def __post_init__(self) -> None:
        if self.value.suffix == GGUF_SUFFIX:
            return

        raise ValueError(
            f"Model path is not a GGUF file: {self.value}. "
            f"Expected a path ending in {GGUF_SUFFIX}."
        )


@dataclass(frozen=True)
class ContextSize:
    """Model context window in tokens. Must be > 0.

    Example:
        context_size = ContextSize(8192)
    """

    value: int

    def __post_init__(self) -> None:
        require_positive_int(self.value, "ContextSize")


@dataclass(frozen=True)
class ThreadCount:
    """CPU thread count for inference. Must be > 0.

    Example:
        thread_count = ThreadCount(8)
    """

    value: int

    def __post_init__(self) -> None:
        require_positive_int(self.value, "ThreadCount")


@dataclass(frozen=True)
class GpuLayerCount:
    """Number of layers offloaded to GPU. 0 means CPU-only. Must be >= 0.

    Example:
        gpu_layer_count = GpuLayerCount(0)
    """

    value: int

    def __post_init__(self) -> None:
        require_non_negative_int(self.value, "GpuLayerCount")
