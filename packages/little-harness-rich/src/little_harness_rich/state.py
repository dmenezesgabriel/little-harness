"""Shared state for the Rich UI plugin."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rich.status import Status

_state = threading.local()


def get_active_status() -> Status | None:
    """Get the currently active status spinner, if any.

    Returns:
        The active Status, or None if no status is active.

    """
    return getattr(_state, "status", None)


def set_active_status(status: Status | None) -> None:
    """Set the currently active status spinner.

    Args:
        status: The Status object, or None to clear.

    """
    _state.status = status
