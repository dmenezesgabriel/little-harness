from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import pytest
from little_harness_llama_cpp.values import (
    ContextSize,
    GpuLayerCount,
    ModelPath,
    ThreadCount,
)


class IntValueObject(Protocol):
    @property
    def value(self) -> int: ...


PositiveIntFactory = Callable[[int], IntValueObject]

POSITIVE_INT_FACTORIES: list[PositiveIntFactory] = [ContextSize, ThreadCount]


class TestPositiveIntegerValues:
    @pytest.mark.parametrize("factory", POSITIVE_INT_FACTORIES)
    def test_accepts_positive_value(self, factory: PositiveIntFactory) -> None:
        # Act / Assert
        assert factory(1).value == 1

    @pytest.mark.parametrize("factory", POSITIVE_INT_FACTORIES)
    def test_rejects_zero_or_negative(self, factory: PositiveIntFactory) -> None:
        # Act / Assert
        with pytest.raises(ValueError, match="is not positive"):
            factory(0)


class TestGpuLayerCount:
    def test_accepts_zero(self) -> None:
        # Act / Assert
        assert GpuLayerCount(0).value == 0

    def test_rejects_negative(self) -> None:
        # Act / Assert
        with pytest.raises(ValueError, match="GpuLayerCount is negative"):
            GpuLayerCount(-1)


class TestModelPath:
    def test_accepts_a_gguf_path_without_requiring_existence(self) -> None:
        # Arrange
        path = Path("models/does-not-exist.gguf")

        # Act / Assert
        assert ModelPath(path).value == path

    def test_rejects_a_non_gguf_path(self) -> None:
        # Act / Assert
        with pytest.raises(ValueError, match="not a GGUF file"):
            ModelPath(Path("models/model.bin"))
