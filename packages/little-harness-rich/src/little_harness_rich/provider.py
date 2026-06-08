"""Entry-point builder for the Rich TUI interactive console.

Registered under the `little_harness.uis` group as `rich`. The core composition
root calls `build()` when this UI is selected (`--ui rich`) and threads the
result through the interactive CLI runner.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from little_harness.application.ports.permission_requester import (
        PermissionRequester,
    )
    from little_harness.presentation.cli.interactive_console import (
        Application,
        InteractiveRunner,
    )
    from little_harness.presentation.cli.repl_command import CommandRegistry

from little_harness_rich.console import RichInteractiveConsole
from little_harness_rich.permission import RichPermissionRequester


def build(
    application: Application,
    registry: CommandRegistry,
) -> InteractiveRunner:
    """Builds a RichInteractiveConsole instance.

    Args:
        application: The agent application runner.
        registry: Registry of slash commands.

    Returns:
        An InteractiveRunner instance.
    """
    return RichInteractiveConsole(application, registry)


def build_permission_requester() -> PermissionRequester:
    """Builds a RichPermissionRequester instance.

    Returns:
        A PermissionRequester instance.
    """
    return RichPermissionRequester()
