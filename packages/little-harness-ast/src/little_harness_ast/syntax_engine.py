"""Port for querying source code by its syntax tree, library-independent."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from little_harness_ast.structural_match import StructuralMatch


class SyntaxEngine(Protocol):
    def find_matches(
        self, source: str, language: str, query: str, /
    ) -> Sequence[StructuralMatch]:
        """Return every node captured as `@match` by `query` in `source`.

        Example:
            matches = engine.find_matches("print(1)", "python", "(call) @match")
        """
        ...
