"""Port for resources that must release native handles when no longer needed."""

from __future__ import annotations

from typing import Protocol


class Closeable(Protocol):
    """Represent a resource that must release native handles."""

    def close(self) -> None:
        """Release any resources the implementation holds.

        Example:
            chat_model.close()

        """
        ...
