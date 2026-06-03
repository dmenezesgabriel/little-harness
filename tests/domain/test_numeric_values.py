from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import pytest

from local_llm.domain.values.numeric_values import (
    ContextSize,
    ElapsedSeconds,
    GpuLayerCount,
    Iteration,
    MaxIterations,
    MaxTokens,
    Temperature,
    ThreadCount,
)


class IntValueObject(Protocol):
    @property
    def value(self) -> int: ...


PositiveIntFactory = Callable[[int], IntValueObject]

POSITIVE_INT_FACTORIES: list[PositiveIntFactory] = [
    MaxTokens,
    MaxIterations,
    Iteration,
    ContextSize,
    ThreadCount,
]


class TestTemperature:
    @pytest.mark.parametrize("value", [0.0, 1.0, 2.0])
    def test_accepts_values_within_range(self, value: float) -> None:
        # Act / Assert
        assert Temperature(value).value == value

    @pytest.mark.parametrize("value", [-0.1, 2.1])
    def test_rejects_values_out_of_range(self, value: float) -> None:
        # Act / Assert
        with pytest.raises(ValueError, match="Temperature out of range"):
            Temperature(value)


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


class TestElapsedSeconds:
    def test_accepts_zero_and_positive(self) -> None:
        # Arrange
        positive = 1.25

        # Act / Assert
        assert ElapsedSeconds(0.0).value == 0.0
        assert ElapsedSeconds(positive).value == positive

    def test_rejects_negative(self) -> None:
        # Act / Assert
        with pytest.raises(ValueError, match="ElapsedSeconds is negative"):
            ElapsedSeconds(-0.5)
