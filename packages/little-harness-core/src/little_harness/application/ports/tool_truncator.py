"""Protocol for truncating tool output before model consumption."""

from __future__ import annotations

from typing import Protocol

from little_harness.domain.values.truncation import TruncationConfig, TruncationResult


class ToolTruncator(Protocol):
    """Strategy for truncating tool output before feeding back to the model.

    Implementations decide which portion of the content to keep (head, tail, etc.)
    and enforce both line and byte limits.

    Example:
        truncator = HeadTruncator()
        result = truncator.truncate("large output", TruncationConfig())

    """

    def truncate(self, content: str, config: TruncationConfig) -> TruncationResult:
        """Return truncated content and metadata."""
        ...
