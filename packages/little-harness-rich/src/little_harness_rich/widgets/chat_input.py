"""Widget for managing chat prompt inputs in the TUI."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Input


class ChatInputWidget(Vertical):
    """A widget providing a wrapped input line for chat messages and commands."""

    def __init__(
        self,
        placeholder: str = "Type your prompt...",
    ) -> None:
        """Initialize the chat input widget.

        Args:
            placeholder: Placeholder text to display.

        """
        super().__init__()
        self.placeholder = placeholder
        self.input = Input(placeholder=placeholder)

    def compose(self) -> ComposeResult:
        """Compose the inner widgets."""
        yield self.input

    @property
    def value(self) -> str:
        """Get the current value of the input field."""
        return self.input.value

    @value.setter
    def value(self, val: str) -> None:
        """Set the value of the input field."""
        self.input.value = val

    def focus(self, scroll_visible: bool = True) -> ChatInputWidget:
        """Focus the input field."""
        self.input.focus(scroll_visible)
        return self
