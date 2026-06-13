"""Widget for displaying tool calls and handling operator approval buttons."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import asyncio

    from little_harness.domain.decision import ToolCall

import json

from rich.syntax import Syntax
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Static


class ToolCallWidget(Vertical):
    """A widget displaying a tool call details and Approve/Reject buttons."""

    def __init__(
        self,
        tool_call: ToolCall,
        future: asyncio.Future[bool] | None = None,
    ) -> None:
        """Initialize the tool call widget.

        Args:
            tool_call: The tool call domain object.
            future: The future to resolve with the user's decision.

        """
        super().__init__()
        self.tool_call = tool_call
        self.future = future
        self._header = Static()
        self._details = Static()
        self._status = Static(id="status-container")

    def compose(self) -> ComposeResult:
        """Compose the tool call widget layout."""
        yield self._header
        yield self._details
        if self.future is not None:
            yield Horizontal(
                Button("Approve (y)", variant="success", id="approve"),
                Button("Reject (n)", variant="error", id="reject"),
                id="buttons-container",
            )
        yield self._status

    def on_mount(self) -> None:
        """Render details on mount."""
        self.add_class("tool-call-card")
        self._render_details()

    def _render_details(self) -> None:
        # Format the arguments as JSON syntax highlighting if possible
        input_value = self.tool_call.tool_input.value
        try:
            parsed = json.loads(input_value)
            formatted = json.dumps(parsed, indent=2)
            renderable = Syntax(formatted, "json", background_color="default")
        except (json.JSONDecodeError, TypeError):
            renderable = Syntax(input_value, "text", background_color="default")

        tool_name = self.tool_call.tool_name.value
        self._header.update(
            f"[bold cyan]Tool Call:[/bold cyan] [yellow]{tool_name}[/yellow]"
        )
        self._details.update(renderable)

    def resolve(self, approved: bool) -> None:
        """Programmatically resolve the tool call approval."""
        if self.future is None or self.future.done():
            return

        self.future.set_result(approved)

        try:
            buttons_container = self.query_one("#buttons-container")
            buttons_container.remove()
        except Exception:  # nosec
            pass

        status_text = (
            "[bold green]✔ Approved[/bold green]"
            if approved
            else "[bold red]❌ Rejected[/bold red]"
        )
        self._status.update(status_text)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle Approve/Reject button clicks.

        Args:
            event: The button pressed event.

        """
        approved = event.button.id == "approve"
        self.resolve(approved)
