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

    def _get_theme_colors(self) -> tuple[str, str]:
        primary_color = "#7AA2F7"
        muted_color = "#8E8E93"
        if not self.is_mounted:
            return primary_color, muted_color
        try:
            app = self.app  # type: ignore[reportUnknownMemberType]
            theme_obj = app.current_theme
            primary_color = str(theme_obj.primary)
            if theme_obj.name == "harness-tokyonight":
                muted_color = "#545C7E"
        except Exception:  # nosec
            pass
        return primary_color, muted_color

    def update_content(self, content: str) -> None:
        """Update the rendered content of the message.

        Args:
            content: The new text content to render.

        """
        self.text_content = content
        if not content:
            self.update("")
            return

        primary_color, muted_color = self._get_theme_colors()
        role_lower = self.role.lower()
        if role_lower == "user":
            text = Text()
            text.append("> ", style=f"bold {primary_color}")
            text.append(content)
            self.update(text)
            return
        if role_lower == "system":
            text = Text()
            text.append("\u2139 ", style=f"bold {muted_color}")
            text.append(content, style=muted_color)
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
