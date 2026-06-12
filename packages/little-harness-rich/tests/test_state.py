# pyright: reportPrivateUsage=false
from __future__ import annotations

from little_harness_rich.app import HarnessTuiApp
from little_harness_rich.state import get_active_app, set_active_app


class FakeTuiApp(HarnessTuiApp):
    """Fake TUI app for testing state."""

    def __init__(self) -> None:
        """Initialize fake app without application core dependencies."""
        pass


def test_get_active_app_returns_none_by_default() -> None:
    """Verify that get_active_app returns None when unset."""
    set_active_app(None)
    assert get_active_app() is None


def test_set_and_get_active_app() -> None:
    """Verify that set_active_app correctly sets and clears active app."""
    app = FakeTuiApp()  # type: ignore[reportAbstractUsage]
    set_active_app(app)
    assert get_active_app() is app
    set_active_app(None)
    assert get_active_app() is None
