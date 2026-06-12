"""Shared state for the Rich UI plugin."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from little_harness_rich.app import HarnessTuiApp


class ActiveAppState:
    """A container for the active TUI application state."""

    def __init__(self) -> None:
        """Initialize the state container."""
        self.app: HarnessTuiApp | None = None


_state = ActiveAppState()


def get_active_app() -> HarnessTuiApp | None:
    """Get the active HarnessTuiApp instance.

    Returns:
        The active HarnessTuiApp, or None if no app is active.

    """
    return _state.app


def set_active_app(app: HarnessTuiApp | None) -> None:
    """Set the active HarnessTuiApp instance.

    Args:
        app: The HarnessTuiApp object, or None to clear.

    """
    _state.app = app
