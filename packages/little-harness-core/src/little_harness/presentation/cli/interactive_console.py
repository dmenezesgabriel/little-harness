"""REPL loop: reads user input, runs agent turns, handles slash commands.

Injected io streams let tests drive the loop with StringIO instead of stdin.
"""

from __future__ import annotations

import contextlib
import readline
import sys
from typing import TYPE_CHECKING, Protocol, TextIO

if TYPE_CHECKING:
    from little_harness.domain.message import ChatMessage
    from little_harness.domain.result import AgentResult

from little_harness.application.ports.skill_loader import SkillLoader
from little_harness.domain.message_history import MessageHistory
from little_harness.domain.values.text_values import Prompt
from little_harness.presentation.cli.repl_command import (
    CommandRegistry,
    ExitReplError,
    build_default_registry,
)


class Application(Protocol):
    """Protocol consumed by the InteractiveConsole.

    Satisfied structurally by ``little_harness.composition.Application``
    so the presentation layer never imports the composition root.
    """

    def build_system_message(self) -> ChatMessage:
        """Build the system message for the conversation."""
        ...

    def run_turn(
        self,
        prompt: Prompt,
        messages: MessageHistory,
    ) -> tuple[AgentResult, MessageHistory]:
        """Run a single agent turn with the given prompt and message history."""
        ...


class InteractiveRunner(Protocol):
    """Protocol defining an interactive UI runner."""

    def start(self) -> str:
        """Start the interactive session and return when the session finishes."""
        ...


class InteractiveConsole:
    """Read-eval-print loop that drives the agent and handles slash commands.

    Injected io streams (output/source) allow tests to drive the loop with
    ``StringIO`` instead of real stdin/stdout.
    """

    def __init__(
        self,
        application: Application,
        output: TextIO | None = None,
        source: TextIO | None = None,
        registry: CommandRegistry | None = None,
        skill_loader: SkillLoader | None = None,
        _initial_messages: MessageHistory | None = None,
    ) -> None:
        """See class docstring for argument descriptions."""
        self._app = application
        self._output = output if output is not None else sys.stdout
        self._source = source if source is not None else sys.stdin
        self._messages: MessageHistory | None = _initial_messages
        self._turn_count = 0
        self._registry: CommandRegistry = (
            registry if registry is not None else build_default_registry()
        )
        self._skill_loader = skill_loader
        self._command_args = ""

    @property
    def registry(self) -> CommandRegistry:
        """Return the command registry used by the console."""
        return self._registry

    @property
    def command_args(self) -> str:
        """Return the remaining text after the slash command name.

        Set by ``_process_command`` before ``execute`` is called.
        """
        return self._command_args

    def clear_history(self) -> None:
        """Clear the conversation history and reset the turn count."""
        self._messages = None
        self._turn_count = 0

    def show_history(self) -> None:
        """Print the conversation history to the output stream."""
        self._output.write(f"Turns: {self._turn_count}\n")
        if self._messages is None:
            self._output.flush()
            return
        for message in self._messages:
            role = message.role.name.capitalize()
            self._output.write(f"  {role}: {message.content.value[:200]}\n")
        self._output.flush()

    def write(self, text: str) -> None:
        """Write text to the output stream and flush."""
        self._output.write(text)
        self._output.flush()

    def list_skills(self) -> str:
        """Return a formatted list of available skills."""
        if self._skill_loader is None:
            return "No skill loader configured.\n"

        skills = self._skill_loader.load_skills()
        if not skills:
            return "No skills loaded.\n"

        lines = ["Available skills:"]
        for skill in skills:
            lines.append(f"  {skill.name.value:<24} {skill.description.value}")
        return "\n".join(lines) + "\n"

    def reload_skills(self) -> str:
        """Re-read skills from disk and return a status message."""
        if self._skill_loader is None:
            return "No skill loader configured.\n"

        skills = self._skill_loader.load_skills()
        count = len(skills)
        return f"Reloaded {count} skill{'s' if count != 1 else ''}.\n"

    def start(self) -> str:
        """Start the REPL loop and return when the session finishes."""
        readline.get_history_length()

        with contextlib.suppress(ExitReplError):
            self._loop()

        return ""

    def _loop(self) -> None:
        while True:
            self._output.write("> ")
            self._output.flush()
            line = self._source.readline()

            if not line:
                self._output.write("\n")
                return

            line = line.strip()

            if not line:
                continue

            if self._process_command(line):
                continue

            self._run_turn(line)

    def _system_messages(self) -> MessageHistory:
        if self._messages is not None:
            return self._messages

        self._messages = MessageHistory().with_message(self._app.build_system_message())
        return self._messages

    def _process_command(self, line: str) -> bool:
        if not line.startswith("/"):
            return False

        parts = line.split(maxsplit=1)
        command = self._registry.get(parts[0])
        self._command_args = parts[1] if len(parts) > 1 else ""

        if command is None:
            self._output.write(f"Unknown command: {line}. Try /help.\n")
            self._output.flush()
            return True

        command.execute(self)
        return True

    def _run_turn(self, text: str) -> None:
        history = self._system_messages()
        result, updated = self._app.run_turn(Prompt(text), history)
        self._messages = updated
        self._turn_count += 1
        self._output.write(result.answer.value + "\n")
        self._output.flush()
