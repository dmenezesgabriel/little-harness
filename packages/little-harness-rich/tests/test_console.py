# pyright: reportPrivateUsage=false
from __future__ import annotations

from io import StringIO

import pytest
from little_harness.domain.message import ChatMessage
from little_harness.domain.message_history import MessageHistory
from little_harness.domain.result import AgentResult
from little_harness.domain.steps import AgentSteps
from little_harness.domain.values.numeric_values import ElapsedSeconds
from little_harness.domain.values.role import ASSISTANT, SYSTEM, USER
from little_harness.domain.values.text_values import (
    MessageContent,
)
from little_harness.domain.values.text_values import (
    Prompt as AgentPrompt,
)
from little_harness.presentation.cli.repl_command import build_default_registry
from little_harness_rich.console import RichInteractiveConsole
from rich.console import Console


class FakeApplication:
    def __init__(self) -> None:
        self.turns: list[tuple[AgentPrompt, MessageHistory]] = []

    def build_system_message(self) -> ChatMessage:
        return ChatMessage(SYSTEM, MessageContent("You are a helpful assistant."))

    def run_turn(
        self, prompt: AgentPrompt, messages: MessageHistory
    ) -> tuple[AgentResult, MessageHistory]:
        self.turns.append((prompt, messages))
        result = AgentResult(
            MessageContent("Mocked agent response"),
            ElapsedSeconds(0.5),
            AgentSteps(),
        )
        updated = messages.with_message(
            ChatMessage(ASSISTANT, MessageContent("Mocked agent response"))
        )
        return result, updated


class TestRichInteractiveConsole:
    def test_loop_executes_turns_and_commands(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        app = FakeApplication()
        output = StringIO()
        console = RichInteractiveConsole(app, build_default_registry())
        # Replace the console instance to capture output and disable color
        console._console = Console(file=output, force_terminal=False, color_system=None)

        inputs = ["Hello", "/help", "/exit"]
        input_iter = iter(inputs)

        def mock_ask(*_args: object, **_kwargs: object) -> str:
            try:
                return next(input_iter)
            except StopIteration as err:
                raise KeyboardInterrupt() from err

        monkeypatch.setattr("rich.prompt.Prompt.ask", mock_ask)

        # Act
        console.start()

        # Assert
        output_text = output.getvalue()
        assert "Agent Interactive Session" in output_text
        assert "Mocked agent response" in output_text
        assert "Available commands:" in output_text
        assert "/help" in output_text
        assert "/exit" in output_text

        # Verify agent was invoked
        assert len(app.turns) == 1
        assert app.turns[0][0].value == "Hello"

    def test_keyboard_interrupt_during_input_exits_gracefully(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        app = FakeApplication()
        output = StringIO()
        console = RichInteractiveConsole(app, build_default_registry())
        console._console = Console(file=output, force_terminal=False, color_system=None)

        def mock_ask(*_args: object, **_kwargs: object) -> str:
            raise KeyboardInterrupt()

        monkeypatch.setattr("rich.prompt.Prompt.ask", mock_ask)

        # Act
        console.start()

        # Assert: exits gracefully and outputs exiting message
        assert "Exiting..." in output.getvalue()

    def test_eof_during_input_exits_gracefully(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        app = FakeApplication()
        output = StringIO()
        console = RichInteractiveConsole(app, build_default_registry())
        console._console = Console(file=output, force_terminal=False, color_system=None)

        def mock_ask(*_args: object, **_kwargs: object) -> str:
            raise EOFError()

        monkeypatch.setattr("rich.prompt.Prompt.ask", mock_ask)

        # Act
        console.start()

        # Assert: exits gracefully and outputs exiting message
        assert "Exiting..." in output.getvalue()

    def test_clear_history_resets_state_and_prints_message(self) -> None:
        # Arrange
        app = FakeApplication()
        output = StringIO()
        console = RichInteractiveConsole(app, build_default_registry())
        console._console = Console(file=output, force_terminal=False, color_system=None)
        console._turn_count = 5

        # Act
        console.clear_history()

        # Assert
        assert console._turn_count == 0
        assert console._messages is None
        assert "History cleared." in output.getvalue()

    def test_show_history_displays_turns_and_messages(self) -> None:
        # Arrange
        app = FakeApplication()
        output = StringIO()
        console = RichInteractiveConsole(app, build_default_registry())
        console._console = Console(file=output, force_terminal=False, color_system=None)

        # Act with no history
        console.show_history()
        assert "Turns: 0" in output.getvalue()

        # Act with history
        output.truncate(0)
        output.seek(0)

        console._turn_count = 1
        console._messages = (
            MessageHistory()
            .with_message(ChatMessage(USER, MessageContent("hello user")))
            .with_message(ChatMessage(ASSISTANT, MessageContent("hello assistant")))
        )

        console.show_history()
        history_text = output.getvalue()
        assert "Turns: 1" in history_text
        assert "User:" in history_text
        assert "hello user" in history_text
        assert "Assistant:" in history_text
        assert "hello assistant" in history_text

    def test_loop_ignores_empty_lines(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Arrange
        app = FakeApplication()
        output = StringIO()
        console = RichInteractiveConsole(app, build_default_registry())
        console._console = Console(file=output, force_terminal=False, color_system=None)

        inputs = ["", "/exit"]
        input_iter = iter(inputs)

        def mock_ask(*_args: object, **_kwargs: object) -> str:
            return next(input_iter)

        monkeypatch.setattr("rich.prompt.Prompt.ask", mock_ask)

        # Act
        console.start()

        # Assert: agent run_turn should not be called since the empty line was ignored
        assert len(app.turns) == 0

    def test_loop_handles_unknown_commands(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        app = FakeApplication()
        output = StringIO()
        console = RichInteractiveConsole(app, build_default_registry())
        console._console = Console(file=output, force_terminal=False, color_system=None)

        inputs = ["/unknown", "/exit"]
        input_iter = iter(inputs)

        def mock_ask(*_args: object, **_kwargs: object) -> str:
            return next(input_iter)

        monkeypatch.setattr("rich.prompt.Prompt.ask", mock_ask)

        # Act
        console.start()

        # Assert
        assert "Unknown command: /unknown" in output.getvalue()
        assert len(app.turns) == 0

    def test_system_messages_cached(self) -> None:
        # Arrange
        app = FakeApplication()
        console = RichInteractiveConsole(app, build_default_registry())

        # Act
        msg1 = console._system_messages()
        msg2 = console._system_messages()

        # Assert
        assert msg1 is msg2
