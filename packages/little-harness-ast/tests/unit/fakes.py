"""Named test double for the syntax engine boundary."""

from __future__ import annotations

from collections.abc import Sequence

from little_harness_ast.structural_match import StructuralMatch


class FakeSyntaxEngine:
    """Returns preset matches (or raises a preset error) and records its calls."""

    def __init__(
        self,
        matches: Sequence[StructuralMatch] = (),
        error: Exception | None = None,
    ) -> None:
        self._matches = matches
        self._error = error
        self.calls: list[tuple[str, str, str]] = []

    def find_matches(
        self, source: str, language: str, query: str
    ) -> Sequence[StructuralMatch]:
        self.calls.append((source, language, query))
        if self._error is not None:
            raise self._error
        return self._matches
