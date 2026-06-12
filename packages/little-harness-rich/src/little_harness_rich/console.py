"""Terminal User Interface runner using the Textual library.

This module implements the interactive console running the Textual TUI.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from little_harness.presentation.cli.repl_command import (
    build_default_registry,
)

from little_harness_rich.app import HarnessTuiApp
from little_harness_rich.state import get_active_app, set_active_app

if TYPE_CHECKING:
    from little_harness.presentation.cli.interactive_console import Application
    from little_harness.presentation.cli.repl_command import CommandRegistry


class RichInteractiveConsole:
    """An interactive session runner that displays agent output in a Textual TUI.

    Conforms structurally to `InteractiveRunner` (entry point runner) and
    `ReplConsole` (slash command context).
    """

    def __init__(
        self,
        application: Application,
        registry: CommandRegistry | None = None,
    ) -> None:
        """Initialize the console runner.

        Args:
            application: The agent application runner.
            registry: Registry of slash commands.

        """
        self._app = application
        self._registry: CommandRegistry = (
            registry if registry is not None else build_default_registry()
        )

    @property
    def registry(self) -> CommandRegistry:
        """The command registry consumed by slash commands."""
        return self._registry

    def clear_history(self) -> None:
        """Clear the conversation history and reset state."""
        app = get_active_app()
        if app is not None:
            app.clear_history()

    def show_history(self) -> None:
        """Print the conversation history in rich formatting."""
        app = get_active_app()
        if app is not None:
            app.show_history()

    def write(self, text: str) -> None:
        """Print text to the terminal."""
        app = get_active_app()
        if app is not None:
            app.write(text)

    def start(self) -> str:
        """Start the interactive session loop.

        Returns:
            An empty string representing normal termination.

        """
        app = HarnessTuiApp(self._app, self._registry)
        set_active_app(app)
        try:
            app.run()
        finally:
            set_active_app(None)

        return ""
