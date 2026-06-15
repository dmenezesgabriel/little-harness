"""Value objects for model reasoning/thinking support."""

from __future__ import annotations

import enum
from dataclasses import dataclass

from little_harness.domain.values.guards import require_positive_int


class ThinkingLevel(enum.StrEnum):
    """How much reasoning the model should expose before the final answer.

    Example:
        level = ThinkingLevel.MEDIUM

    """

    OFF = "off"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class ThinkingBudget:
    """Maximum tokens the model may use for reasoning. Must be > 0.

    Example:
        budget = ThinkingBudget(2048)

    """

    value: int

    def __post_init__(self) -> None:
        """Validate that the budget is a positive integer."""
        require_positive_int(self.value, "ThinkingBudget")


@dataclass(frozen=True)
class ThinkingContent:
    """The reasoning/thinking tokens a model produced internally.

    Separated from visible `MessageContent` so consumers can display or filter
    the chain of thought without affecting the visible answer.

    Example:
        thought = ThinkingContent("Let me calculate 144 / 12...")

    """

    value: str
