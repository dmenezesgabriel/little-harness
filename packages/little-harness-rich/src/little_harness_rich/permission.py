"""Permission requester using Rich prompts."""

from __future__ import annotations

from little_harness.application.ports.permission_requester import PermissionRequester
from little_harness.domain.decision import ToolCall
from rich.console import Console
from rich.prompt import Confirm

from little_harness_rich.state import get_active_status


class RichPermissionRequester(PermissionRequester):
    """Prompts the operator to approve a sensitive tool call using Rich."""

    def __init__(self, console: Console | None = None) -> None:
        """Initializes the permission requester.

        Args:
            console: The Rich console to use for prompting.
        """
        self._console = console or Console()

    def request_approval(self, call: ToolCall, /) -> bool:
        """Prompts the operator to approve a tool call.

        Args:
            call: The tool call to approve.

        Returns:
            True if the operator approved the call, False otherwise.
        """
        status = get_active_status()
        if status is not None:
            status.stop()

        try:
            return Confirm.ask(
                f"Allow tool [cyan]{call.tool_name.value!r}[/cyan] to run with input "
                f"[yellow]{call.tool_input.value!r}[/yellow]?",
                console=self._console,
                default=False,
            )
        finally:
            if status is not None:
                status.start()
