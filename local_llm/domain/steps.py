"""First-class collection of recorded agent steps."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from local_llm.domain.step import AgentStep


@dataclass(frozen=True)
class AgentSteps:
    """An immutable, ordered trace of loop iterations. Grow with `with_step`.

    Example:
        steps = AgentSteps().with_step(first_step)
    """

    _steps: tuple[AgentStep, ...] = ()

    def with_step(self, step: AgentStep) -> AgentSteps:
        return AgentSteps((*self._steps, step))

    def is_empty(self) -> bool:
        return len(self._steps) == 0

    def __iter__(self) -> Iterator[AgentStep]:
        return iter(self._steps)

    def __len__(self) -> int:
        return len(self._steps)
