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

from little_harness.domain.message_history import MessageHistory
from little_harness.domain.values.text_values import Prompt
from little_harness.presentation.cli.repl_command import (
    CommandRegistry,
    _ExitReplError,
    build_default_registry,
)


class Application(Protocol):
    """Protocol consumed by the InteractiveConsole.

    Satisfied structurally by ``little_harness.composition.Application``
    so the presentation layer never imports the composition root.
    """

    def build_system_message(self) -> ChatMessage: ...

    def run_turn(
        self,
        prompt: Prompt,
        messages: MessageHistory,
    ) -> tuple[AgentResult, MessageHistory]: ...


class InteractiveConsole:
    def __init__(
        self,
        application: Application,
        output: TextIO | None = None,
        source: TextIO | None = None,
        registry: CommandRegistry | None = None,
    ) -> None:
        self._app = application
        self._output = output if output is not None else sys.stdout
        self._source = source if source is not None else sys.stdin
        self._messages: MessageHistory | None = None
        self._turn_count = 0
        self._registry: CommandRegistry = (
            registry if registry is not None else build_default_registry()
        )

    @property
    def registry(self) -> CommandRegistry:
        return self._registry

    def clear_history(self) -> None:
        self._messages = None
        self._turn_count = 0

    def show_history(self) -> None:
        self._output.write(f"Turns: {self._turn_count}\n")
        if self._messages is None:
            self._output.flush()
            return
        for message in self._messages:
            role = message.role.name.capitalize()
            self._output.write(f"  {role}: {message.content.value[:200]}\n")
        self._output.flush()

    def write(self, text: str) -> None:
        self._output.write(text)
        self._output.flush()

    def start(self) -> str:
        readline.get_history_length()

        with contextlib.suppress(_ExitReplError):
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

        command = self._registry.get(line)

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
