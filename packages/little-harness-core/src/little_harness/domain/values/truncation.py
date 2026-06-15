"""Value objects for tool-output truncation limits and results."""

from __future__ import annotations

from dataclasses import dataclass

from little_harness.domain.values.guards import require_positive_int


@dataclass(frozen=True)
class TruncationConfig:
    """Configuration for tool output truncation limits.

    Example:
        config = TruncationConfig(max_lines=500, max_bytes=10240)

    """

    max_lines: int = 2000
    max_bytes: int = 51200

    def __post_init__(self) -> None:
        """Validate that both limits are positive integers."""
        require_positive_int(self.max_lines, "MaxLines")
        require_positive_int(self.max_bytes, "MaxBytes")


@dataclass(frozen=True)
class TruncationResult:
    """Outcome of truncating tool output content.

    Example:
        result = TruncationResult(
            content="line1", truncated=True,
            truncated_by="lines", total_lines=100, total_bytes=5000,
            output_lines=1, output_bytes=5,
        )

    """

    content: str
    truncated: bool
    truncated_by: str | None
    total_lines: int
    total_bytes: int
    output_lines: int
    output_bytes: int
