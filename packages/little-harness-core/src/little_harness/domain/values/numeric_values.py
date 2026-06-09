"""Bounded numeric value objects for sampling and the agent loop."""

from __future__ import annotations

from dataclasses import dataclass

from little_harness.domain.values.guards import (
    require_non_negative,
    require_positive_int,
    require_within,
)


@dataclass(frozen=True)
class Temperature:
    """Sampling temperature, constrained to 0.0..2.0.

    Example:
        temperature = Temperature(0.0)

    """

    value: float

    def __post_init__(self) -> None:
        """Validate that temperature is within 0.0..2.0."""
        require_within(self.value, 0.0, 2.0, "Temperature")


@dataclass(frozen=True)
class MaxTokens:
    """Maximum number of generated tokens. Must be > 0.

    Example:
        max_tokens = MaxTokens(512)

    """

    value: int

    def __post_init__(self) -> None:
        """Validate that max tokens is positive."""
        require_positive_int(self.value, "MaxTokens")


@dataclass(frozen=True)
class MaxIterations:
    """Upper bound on agent loop iterations. Must be > 0.

    Example:
        max_iterations = MaxIterations(5)

    """

    value: int

    def __post_init__(self) -> None:
        """Validate that max iterations is positive."""
        require_positive_int(self.value, "MaxIterations")


@dataclass(frozen=True)
class Iteration:
    """A 1-based agent loop iteration counter. Must be >= 1.

    Example:
        iteration = Iteration(1)

    """

    value: int

    def __post_init__(self) -> None:
        """Validate that iteration is positive."""
        require_positive_int(self.value, "Iteration")


@dataclass(frozen=True)
class TopP:
    """Nucleus sampling probability threshold. 1.0 disables truncation.

    Example:
        top_p = TopP(0.95)

    """

    value: float

    def __post_init__(self) -> None:
        """Validate that top-p is within 0.0..1.0."""
        require_within(self.value, 0.0, 1.0, "TopP")


@dataclass(frozen=True)
class RepeatPenalty:
    """Token repetition penalty. 1.0 means no penalty.

    Example:
        repeat_penalty = RepeatPenalty(1.0)

    """

    value: float

    def __post_init__(self) -> None:
        """Validate that repeat penalty is within 0.0..2.0."""
        require_within(self.value, 0.0, 2.0, "RepeatPenalty")


@dataclass(frozen=True)
class ElapsedSeconds:
    """Wall-clock duration of an agent run in seconds. Must be >= 0.

    Example:
        elapsed = ElapsedSeconds(1.25)

    """

    value: float

    def __post_init__(self) -> None:
        """Validate that elapsed seconds is non-negative."""
        require_non_negative(self.value, "ElapsedSeconds")
