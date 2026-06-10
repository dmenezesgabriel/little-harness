"""ReplCommand protocol, registry, and built-in slash commands.

Built-in commands (ClearCommand, ExitCommand, HelpCommand, HistoryCommand) are
registered by default. External packages can add commands via the entry point
group ``little_harness.repl_commands``; each builder returns a ``ReplCommand``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from operator import attrgetter
from typing import Protocol

logger = logging.getLogger(__name__)


class CommandRegistry:
    """Name-to-command mapping that rejects duplicate names at construction.

    The registry tracks overrides: when a second command registers the same
    name, a warning is logged and the override is recorded for debugging.
    """

    def __init__(self) -> None:
        """See class docstring for argument descriptions."""
        self._index: dict[str, ReplCommand] = {}
        self._sources: dict[str, str] = {}
        self._overrides: dict[str, str] = {}

    def add(self, command: ReplCommand, source: str) -> None:
        """Register *command* under its name and each alias."""
        for alias in (command.name, *command.aliases):
            self._register_alias(alias, command, source)

    def _register_alias(self, alias: str, command: ReplCommand, source: str) -> None:
        key = f"/{alias}"
        if key in self._index:
            self._track_override(key, source)
        self._index[key] = command
        self._sources[key] = source

    def _track_override(self, key: str, source: str) -> None:
        previous = self._sources[key]
        logger.warning(
            "REPL command %r overridden by %s (was %s)",
            key,
            source,
            previous,
        )
        self._overrides[key] = f"{previous} \u2192 {source}"

    def get(self, raw: str) -> ReplCommand | None:
        """Look up a command by its /-prefixed name (case-insensitive)."""
        return self._index.get(raw.lower())

    @property
    def overrides(self) -> dict[str, str]:
        """Return a copy of the override tracking dict."""
        return dict(self._overrides)

    def __iter__(self) -> Iterator[ReplCommand]:
        """Yield each unique command object (once per name, not per alias)."""
        seen: set[int] = set()
        for command in self._index.values():
            command_id = id(command)
            if command_id not in seen:
                seen.add(command_id)
                yield command


class ReplConsole(Protocol):
    """Protocol defining the operations a REPL command can invoke on the console."""

    @property
    def registry(self) -> CommandRegistry:
        """Return the command registry."""
        ...

    def clear_history(self) -> None:
        """Clear the conversation history."""
        ...

    def show_history(self) -> None:
        """Show the conversation history."""
        ...

    def write(self, text: str) -> None:
        """Write text to the console output."""
        ...


class ReplCommand(Protocol):
    """Structural protocol for a REPL slash command."""

    name: str
    aliases: tuple[str, ...]
    description: str

    def execute(self, console: ReplConsole, /) -> None:
        """Execute the command against the given console."""
        ...


class ExitReplError(Exception):
    """Signal to exit the REPL loop cleanly.

    Raised by ExitCommand.execute and caught by InteractiveConsole.start.
    """


# ── Built-in commands ───────────────────────────────────────────────────────


class ClearCommand:
    """Built-in ``/clear`` slash command that resets conversation history."""

    name = "clear"
    aliases: tuple[str, ...] = ()
    description = "Clear conversation history"

    def execute(self, console: ReplConsole, /) -> None:
        """Clear the conversation history."""
        console.clear_history()


class ExitCommand:
    """Built-in ``/exit`` slash command that terminates the REPL loop."""

    name = "exit"
    aliases: tuple[str, ...] = ("quit",)
    description = "Exit the interactive session"

    def execute(self, _console: ReplConsole, /) -> None:
        """Exit the interactive session."""
        raise ExitReplError()


class HelpCommand:
    """Built-in ``/help`` slash command that lists available commands."""

    name = "help"
    aliases: tuple[str, ...] = ()
    description = "Show this help message"

    def execute(self, console: ReplConsole, /) -> None:
        """Show available commands and their descriptions."""
        lines = ["Available commands:"]
        for command in sorted(console.registry, key=attrgetter("name")):
            names = f"/{command.name}"
            for alias in command.aliases:
                names += f", /{alias}"
            lines.append(f"  {names:<24} {command.description}")
        console.write("\n".join(lines) + "\n")


class HistoryCommand:
    """Built-in ``/history`` slash command that shows conversation history."""

    name = "history"
    aliases: tuple[str, ...] = ()
    description = "Show conversation history"

    def execute(self, console: ReplConsole, /) -> None:
        """Show the conversation history."""
        console.show_history()


def builtin_commands() -> list[ReplCommand]:
    """Return the default set of built-in slash commands."""
    return [ClearCommand(), ExitCommand(), HelpCommand(), HistoryCommand()]


def build_default_registry() -> CommandRegistry:
    """Create a CommandRegistry with all built-in commands registered."""
    registry = CommandRegistry()
    for command in builtin_commands():
        registry.add(command, "built-in")
    return registry
