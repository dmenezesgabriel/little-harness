"""First-class collection of recorded agent steps."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from little_harness.domain.step import AgentStep


@dataclass(frozen=True)
class AgentSteps:
    """An immutable, ordered trace of loop iterations. Grow with `with_step`.

    Example:
        steps = AgentSteps().with_step(first_step)

    """

    _steps: tuple[AgentStep, ...] = ()

    def with_step(self, step: AgentStep) -> AgentSteps:
        """Return a new collection with `step` appended."""
        return AgentSteps((*self._steps, step))

    def is_empty(self) -> bool:
        """Return `True` when no steps have been recorded."""
        return len(self._steps) == 0

    def __iter__(self) -> Iterator[AgentStep]:
        """Yield each step in order."""
        return iter(self._steps)

    def __len__(self) -> int:
        """Return the number of steps."""
        return len(self._steps)
