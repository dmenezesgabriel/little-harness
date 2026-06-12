"""Widget for displaying individual chat messages in the conversation stream."""

from __future__ import annotations

from rich.markdown import Markdown
from rich.text import Text
from textual.widgets import Static


class ChatMessageWidget(Static):
    """A widget displaying a single message in the conversation stream."""

    def __init__(
        self,
        role: str,
        content: str,
    ) -> None:
        """Initialize the message widget.

        Args:
            role: The role (e.g. 'user', 'assistant', 'system').
            content: The text content of the message.

        """
        super().__init__()
        self.role = role
        self.text_content = content

    def on_mount(self) -> None:
        """Set style classes on mount."""
        self.add_class("chat-bubble")
        self.add_class(self.role.lower())
        self.update_content(self.text_content)

    def update_content(self, content: str) -> None:
        """Update the rendered content of the message.

        Args:
            content: The new text content to render.

        """
        self.text_content = content
        if not content:
            self.update("")
            return

        role_lower = self.role.lower()
        if role_lower == "user":
            text = Text()
            text.append("> ", style="bold #0A84FF")
            text.append(content)
            self.update(text)
            return
        if role_lower == "system":
            text = Text()
            text.append("\u2139 ", style="bold #8E8E93")
            text.append(content, style="#8E8E93")
            self.update(text)
            return
        # Render markdown content cleanly
        self.update(Markdown(content))

    @classmethod
    def user(cls, content: str) -> ChatMessageWidget:
        """Create a user message widget.

        Args:
            content: The message content.

        Returns:
            A ChatMessageWidget instance.

        """
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str = "") -> ChatMessageWidget:
        """Create an assistant message widget.

        Args:
            content: The message content.

        Returns:
            A ChatMessageWidget instance.

        """
        return cls(role="assistant", content=content)

    @classmethod
    def system(cls, content: str) -> ChatMessageWidget:
        """Create a system message widget.

        Args:
            content: The message content.

        Returns:
            A ChatMessageWidget instance.

        """
        return cls(role="system", content=content)
