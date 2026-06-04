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
        require_within(self.value, 0.0, 2.0, "Temperature")


@dataclass(frozen=True)
class MaxTokens:
    """Maximum number of generated tokens. Must be > 0.

    Example:
        max_tokens = MaxTokens(512)
    """

    value: int

    def __post_init__(self) -> None:
        require_positive_int(self.value, "MaxTokens")


@dataclass(frozen=True)
class MaxIterations:
    """Upper bound on agent loop iterations. Must be > 0.

    Example:
        max_iterations = MaxIterations(5)
    """

    value: int

    def __post_init__(self) -> None:
        require_positive_int(self.value, "MaxIterations")


@dataclass(frozen=True)
class Iteration:
    """A 1-based agent loop iteration counter. Must be >= 1.

    Example:
        iteration = Iteration(1)
    """

    value: int

    def __post_init__(self) -> None:
        require_positive_int(self.value, "Iteration")


@dataclass(frozen=True)
class ElapsedSeconds:
    """Wall-clock duration of an agent run in seconds. Must be >= 0.

    Example:
        elapsed = ElapsedSeconds(1.25)
    """

    value: float

    def __post_init__(self) -> None:
        require_non_negative(self.value, "ElapsedSeconds")
