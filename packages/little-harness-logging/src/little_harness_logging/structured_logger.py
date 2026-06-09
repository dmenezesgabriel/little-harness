"""Port for structured (JSON) logging, owned by this project."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol


class StructuredLogger(Protocol):
    """Protocol for emitting structured (JSON) log records."""

    def log(self, event: str, fields: Mapping[str, object]) -> None:
        """Emit a structured log record for an event with typed fields.

        Example:
            logger.log("tool_invoked", {"tool": "calculator", "ok": True})

        """
        ...
