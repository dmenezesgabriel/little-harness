"""Widget for displaying agent thinking/reasoning steps in a collapsible panel."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Collapsible, Static


class ReasoningBlockWidget(Widget):
    """A widget that displays the agent's thinking process."""

    SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")

    def __init__(self) -> None:
        """Initialize the reasoning block widget."""
        super().__init__()
        self._text_widget = Static()
        self._collapsible: Collapsible | None = None
        self._accumulated_text = ""
        self._spinner_index = 0
        self._timer = None
        self._completed = False

    def compose(self) -> ComposeResult:
        """Compose the collapsible widget on request."""
        collapsible = Collapsible(
            self._text_widget,
            title="Thinking...",
            collapsed=False,
        )
        self._collapsible = collapsible
        yield collapsible

    def on_mount(self) -> None:
        """Start the spinner animation timer on mount."""
        self._timer = self.set_interval(0.1, self._update_spinner)

    def _update_spinner(self) -> None:
        """Update the spinner frame in the collapsible title."""
        if self._completed:
            return
        frame = self.SPINNER_FRAMES[self._spinner_index]
        self._spinner_index = (self._spinner_index + 1) % len(self.SPINNER_FRAMES)
        self.title = f"{frame} Thinking..."

    @property
    def title(self) -> str:
        """Get the title of the collapsible block."""
        return self._collapsible.title if self._collapsible is not None else ""

    @title.setter
    def title(self, val: str) -> None:
        """Set the title of the collapsible block."""
        if self._collapsible is not None:
            self._collapsible.title = val

    @property
    def collapsed(self) -> bool:
        """Get the collapsed state."""
        return self._collapsible.collapsed if self._collapsible is not None else False

    @collapsed.setter
    def collapsed(self, val: bool) -> None:
        """Set the collapsed state."""
        if self._collapsible is not None:
            self._collapsible.collapsed = val

    def update_reasoning(self, text: str) -> None:
        """Update the reasoning text.

        Args:
            text: The latest reasoning monologue text.

        """
        self._accumulated_text = text
        self._text_widget.update(text)

    def append_reasoning(self, text: str) -> None:
        """Append reasoning text chunk.

        Args:
            text: The text chunk to append.

        """
        self._accumulated_text += text
        self._text_widget.update(self._accumulated_text)

    def complete(self) -> None:
        """Mark the reasoning as complete, updating title and collapsing."""
        self._completed = True
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        self.title = "Thought Process (Done)"
        self.collapsed = True
