"""Tests for thinking/reasoning domain value objects."""

from __future__ import annotations

import pytest
from little_harness.domain.values.thinking import (
    ThinkingBudget,
    ThinkingContent,
    ThinkingLevel,
)


class TestThinkingLevel:
    def test_enum_values(self) -> None:
        assert ThinkingLevel.OFF.value == "off"
        assert ThinkingLevel.LOW.value == "low"
        assert ThinkingLevel.MEDIUM.value == "medium"
        assert ThinkingLevel.HIGH.value == "high"


class TestThinkingBudget:
    def test_accepts_positive_int(self) -> None:
        budget = ThinkingBudget(1024)
        assert budget.value == 1024

    @pytest.mark.parametrize("value", [0, -1])
    def test_rejects_non_positive(self, value: int) -> None:
        with pytest.raises(ValueError, match="ThinkingBudget is not positive"):
            ThinkingBudget(value)


class TestThinkingContent:
    def test_accepts_text(self) -> None:
        content = ThinkingContent("Let me think step by step...")
        assert content.value == "Let me think step by step..."

    def test_accepts_empty_text(self) -> None:
        content = ThinkingContent("")
        assert content.value == ""

    def test_repr_includes_value(self) -> None:
        content = ThinkingContent("reasoning")
        assert "reasoning" in repr(content)
