"""Filesystem path to a local GGUF model file."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
