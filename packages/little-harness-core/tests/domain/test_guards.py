"""Direct tests for the shared validation guards.

Guards are a public API of core reused by provider plugins, so they are tested
here directly rather than only through the value objects that call them.
"""

from __future__ import annotations

import pytest
from little_harness.domain.values.guards import (
    require_non_empty_text,
    require_non_negative,
    require_non_negative_int,
    require_positive_int,
    require_within,
)


class TestRequireNonEmptyText:
    def test_returns_the_trimmed_value(self) -> None:
        assert require_non_empty_text("  calc  ", "ToolName") == "calc"

    @pytest.mark.parametrize("value", ["", "   "])
    def test_rejects_blank_text(self, value: str) -> None:
        with pytest.raises(ValueError, match="ToolName is empty"):
            require_non_empty_text(value, "ToolName")


class TestRequirePositiveInt:
    def test_returns_a_positive_value(self) -> None:
        positive = 3
        assert require_positive_int(positive, "MaxTokens") == positive

    @pytest.mark.parametrize("value", [0, -1])
    def test_rejects_non_positive(self, value: int) -> None:
        with pytest.raises(ValueError, match="MaxTokens is not positive"):
            require_positive_int(value, "MaxTokens")


class TestRequireNonNegativeInt:
    @pytest.mark.parametrize("value", [0, 7])
    def test_returns_a_non_negative_value(self, value: int) -> None:
        assert require_non_negative_int(value, "GpuLayerCount") == value

    def test_rejects_negative(self) -> None:
        with pytest.raises(ValueError, match="GpuLayerCount is negative"):
            require_non_negative_int(-1, "GpuLayerCount")


class TestRequireWithin:
    @pytest.mark.parametrize("value", [0.0, 1.0, 2.0])
    def test_returns_a_value_in_range(self, value: float) -> None:
        assert require_within(value, 0.0, 2.0, "Temperature") == value

    @pytest.mark.parametrize("value", [-0.1, 2.1])
    def test_rejects_out_of_range(self, value: float) -> None:
        with pytest.raises(ValueError, match="Temperature out of range"):
            require_within(value, 0.0, 2.0, "Temperature")


class TestRequireNonNegative:
    @pytest.mark.parametrize("value", [0.0, 1.25])
    def test_returns_a_non_negative_value(self, value: float) -> None:
        assert require_non_negative(value, "ElapsedSeconds") == value

    def test_rejects_negative(self) -> None:
        with pytest.raises(ValueError, match="ElapsedSeconds is negative"):
            require_non_negative(-0.5, "ElapsedSeconds")
