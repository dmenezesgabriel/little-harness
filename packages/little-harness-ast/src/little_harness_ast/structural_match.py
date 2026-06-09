"""Value object describing one syntax-tree match, independent of any library.

Carries both a human-facing line span (for search results) and a byte span (for
precise, structure-aware edits), so both the grep and edit tools speak the same
vendor-free language.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StructuralMatch:
    """A matched node: its 1-based line span, its byte span, and its text.

    Example:
        match = StructuralMatch(2, 2, 6, 17, "print('hi')")

    """

    start_line: int
    end_line: int
    start_byte: int
    end_byte: int
    text: str

    def location(self) -> str:
        """Return a human-readable line reference (e.g. ``line 3`` or ``lines 3-5``)."""
        if self.start_line == self.end_line:
            return f"line {self.start_line}"

        return f"lines {self.start_line}-{self.end_line}"
