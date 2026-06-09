# pyright: reportPrivateUsage=false
from __future__ import annotations

from little_harness_rich.state import _state, get_active_status, set_active_status
from rich.console import Console
from rich.status import Status


def test_get_active_status_returns_none_when_unset() -> None:
    set_active_status(None)

    # Delete the attribute to test the default getattr behavior
    if hasattr(_state, "status"):
        delattr(_state, "status")

    assert get_active_status() is None


def test_set_and_get_active_status() -> None:
    console = Console(force_terminal=True)
    status = Status("thinking", console=console)
    set_active_status(status)
    assert get_active_status() is status
    set_active_status(None)
    assert get_active_status() is None
