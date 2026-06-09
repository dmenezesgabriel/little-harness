"""Terminal User Interface runner using the Rich library.

This module implements the interactive console using rich components like Panel,
status spinners, and markdown rendering.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

if TYPE_CHECKING:
    from little_harness.presentation.cli.interactive_console import Application
    from little_harness.presentation.cli.repl_command import CommandRegistry

from little_harness.domain.message_history import MessageHistory
from little_harness.domain.values.role import ASSISTANT, USER
from little_harness.domain.values.text_values import Prompt as AgentPrompt
from little_harness.presentation.cli.repl_command import (
    ExitReplError,
    build_default_registry,
)

from little_harness_rich.state import set_active_status


class RichInteractiveConsole:
    """An interactive session runner that displays agent output in color/markdown.

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
        self._console = Console()
        self._messages: MessageHistory | None = None
        self._turn_count = 0
        self._registry: CommandRegistry = (
            registry if registry is not None else build_default_registry()
        )

    @property
    def registry(self) -> CommandRegistry:
        """The command registry consumed by slash commands."""
        return self._registry

    def clear_history(self) -> None:
        """Clear the message history and reset turn counts."""
        self._messages = None
        self._turn_count = 0
        self._console.print("[yellow]History cleared.[/yellow]")

    def show_history(self) -> None:
        """Print the conversation history in rich formatting."""
        self._console.print(f"Turns: {self._turn_count}")
        if self._messages is None:
            return
        for message in self._messages:
            role = message.role.name.capitalize()
            role_style = (
                "green"
                if message.role == USER
                else "blue"
                if message.role == ASSISTANT
                else "magenta"
            )
            self._console.print(f"[bold {role_style}]{role}:[/bold {role_style}]")
            self._console.print(Markdown(message.content.value))
            self._console.print()

    def write(self, text: str) -> None:
        """Print text to the terminal."""
        self._console.print(text)

    def start(self) -> str:
        """Start the interactive session loop.

        Returns:
            An empty string representing normal termination.

        """
        self._console.print(
            Panel(
                "[bold green]Welcome to the Little Harness Agent![/bold green]\n"
                "Type your prompt to start, or [cyan]/help[/cyan] "
                "for available commands.",
                title="Agent Interactive Session",
                border_style="cyan",
            )
        )

        with contextlib.suppress(ExitReplError):
            self._loop()

        return ""

    def _loop(self) -> None:
        while True:
            try:
                line = Prompt.ask(">")
            except (KeyboardInterrupt, EOFError):
                self._console.print("\n[yellow]Exiting...[/yellow]")
                break  # pragma: no mutate

            line = line.strip()
            if not line:
                continue

            if self._process_command(line):
                continue

            self._run_turn(line)

    def _process_command(self, line: str) -> bool:
        if not line.startswith("/"):
            return False

        command = self._registry.get(line)
        if command is None:
            self._console.print(
                f"[red]Unknown command:[/red] {line}. Try [cyan]/help[/cyan]."
            )
            return True

        command.execute(self)
        return True

    def _system_messages(self) -> MessageHistory:
        if self._messages is not None:
            return self._messages

        self._messages = MessageHistory().with_message(self._app.build_system_message())
        return self._messages

    def _run_turn(self, text: str) -> None:
        history = self._system_messages()

        status = self._console.status("[bold blue]Agent is thinking...[/bold blue]")
        set_active_status(status)
        try:
            with status:
                result, updated = self._app.run_turn(AgentPrompt(text), history)
        finally:
            set_active_status(None)

        self._messages = updated
        self._turn_count += 1

        self._console.print()
        self._console.print(Markdown(result.answer.value))
        self._console.print()
