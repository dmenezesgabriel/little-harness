"""Named test double for the ripgrep search boundary."""

from __future__ import annotations

from collections.abc import Sequence

from little_harness_ripgrep.ripgrep_search import RipgrepOutcome


class FakeRipgrepSearch:
    """Returns a preset outcome and records the arguments and timeout received."""

    def __init__(self, outcome: RipgrepOutcome) -> None:
        self._outcome = outcome
        self.argument_calls: list[Sequence[str]] = []
        self.timeouts: list[float] = []

    def run(self, arguments: Sequence[str], timeout_seconds: float) -> RipgrepOutcome:
        self.argument_calls.append(arguments)
        self.timeouts.append(timeout_seconds)
        return self._outcome
