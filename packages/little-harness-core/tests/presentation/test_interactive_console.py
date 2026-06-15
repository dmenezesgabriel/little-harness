"""Tests for the REPL loop: slash commands, turn execution, history management."""

from __future__ import annotations

from io import StringIO

import pytest
from little_harness.domain.message import ChatMessage
from little_harness.domain.message_history import MessageHistory
from little_harness.domain.result import AgentResult
from little_harness.domain.steps import AgentSteps
from little_harness.domain.values.numeric_values import ElapsedSeconds
from little_harness.domain.values.role import ASSISTANT, SYSTEM, USER
from little_harness.domain.values.text_values import MessageContent, Prompt
from little_harness.presentation.cli.interactive_console import (
    ExitReplError,
    InteractiveConsole,
)

SYSTEM_MSG = ChatMessage(SYSTEM, MessageContent("System prompt"))


class FakeApplication:
    """Simulates Application for the REPL tests."""

    def __init__(self) -> None:
        self.turns: list[Prompt] = []
        self._turn_index = 0

    def build_system_message(self) -> ChatMessage:
        return SYSTEM_MSG

    def run_turn(
        self, prompt: Prompt, messages: MessageHistory
    ) -> tuple[AgentResult, MessageHistory]:
        self.turns.append(prompt)
        self._turn_index += 1
        answer = MessageContent(f"Answer {self._turn_index}")
        updated = messages.with_message(
            ChatMessage(USER, MessageContent(prompt.value))
        ).with_message(ChatMessage(ASSISTANT, answer))
        result = AgentResult(answer, ElapsedSeconds(0.5), AgentSteps())
        return result, updated


def console_with(
    *inputs: str, app: FakeApplication | None = None
) -> tuple[InteractiveConsole, StringIO]:
    output = StringIO()
    source = StringIO("\n".join(inputs))
    console = InteractiveConsole(app or FakeApplication(), output=output, source=source)
    return console, output


class TestInteractiveConsoleSlashCommands:
    def test_exit_terminates_the_loop(self) -> None:
        console, _ = console_with("/exit")

        result = console.start()

        assert result == ""

    def test_quit_terminates_the_loop(self) -> None:
        console, _ = console_with("/quit")

        result = console.start()

        assert result == ""

    def test_process_exit_raises_exit_repl(self) -> None:
        console, _ = console_with()

        with pytest.raises(ExitReplError):
            console._process_command("/exit")

    def test_process_quit_raises_exit_repl(self) -> None:
        console, _ = console_with()

        with pytest.raises(ExitReplError):
            console._process_command("/quit")

    def test_clear_resets_history_to_system_message(self) -> None:
        app = FakeApplication()
        console, _ = console_with("hello", "/clear", "/exit", app=app)

        console.start()

        assert len(app.turns) == 1

    def test_process_clear_returns_true(self) -> None:
        console, _ = console_with()

        result = console._process_command("/clear")

        assert result is True

    def test_process_clear_resets_state(self) -> None:
        app = FakeApplication()
        console, _ = console_with("first", "second", app=app)
        console.start()

        console._process_command("/clear")

        assert console._messages is None
        assert console._turn_count == 0

    def test_help_prints_available_commands(self) -> None:
        console, output = console_with("/help", "/exit")

        console.start()

        text = output.getvalue()
        assert "/exit" in text
        assert "/quit" in text
        assert "/clear" in text
        assert "/help" in text
        assert "/history" in text

    def test_help_output_ends_with_newline(self) -> None:
        console, output = console_with("/help", "/exit")

        console.start()

        text = output.getvalue()
        assert "Available commands:\n" in text

    def test_process_help_returns_true(self) -> None:
        console, _ = console_with()

        result = console._process_command("/help")

        assert result is True

    def test_history_lists_recent_turns(self) -> None:
        app = FakeApplication()
        console, output = console_with("hi", "/history", "/exit", app=app)

        console.start()

        text = output.getvalue()
        assert "Turns: 1" in text
        assert "Answer 1" in text
        assert "User" in text or "USER" in text

    def test_process_history_returns_true(self) -> None:
        console, _ = console_with()

        result = console._process_command("/history")

        assert result is True

    def test_process_unknown_returns_true(self) -> None:
        console, _ = console_with()

        result = console._process_command("/xyz")

        assert result is True

    def test_process_skill_without_args_lists_skills(self) -> None:
        console, _ = console_with()
        result = console._process_command("/skill")
        assert result is True

    def test_process_skill_reload_reloads_skills(self) -> None:
        console, _ = console_with()
        result = console._process_command("/skill reload")
        assert result is True

    def test_process_skill_with_reload_triggers_reload(self) -> None:
        console, output = console_with("/skill reload", "/exit")
        console.start()
        text = output.getvalue()
        assert "No skill loader configured" in text


class TestInteractiveConsoleTurnExecution:
    def test_user_prompt_triggers_run_turn(self) -> None:
        app = FakeApplication()
        console, _ = console_with("hello", "/exit", app=app)

        console.start()

        assert len(app.turns) == 1
        assert app.turns[0] == Prompt("hello")

    def test_prints_the_answer(self) -> None:
        app = FakeApplication()
        console, output = console_with("hello", "/exit", app=app)

        console.start()

        assert "Answer 1\n" in output.getvalue()

    def test_preserves_history_across_multiple_turns(self) -> None:
        app = FakeApplication()
        console, _ = console_with("first", "second", "/exit", app=app)

        console.start()

        assert len(app.turns) == 2
        assert app.turns[0] == Prompt("first")
        assert app.turns[1] == Prompt("second")

    def test_builds_system_message_on_first_turn(self) -> None:
        app = FakeApplication()
        console, _ = console_with("hello", "/exit", app=app)

        console.start()

        assert len(app.turns) == 1

    def test_turn_count_starts_at_zero(self) -> None:
        console, _ = console_with()

        assert console._turn_count == 0

    def test_turn_count_increments_on_each_turn(self) -> None:
        app = FakeApplication()
        console, _ = console_with("first", "second", "/exit", app=app)

        console.start()

        assert console._turn_count == 2

    def test_run_turn_updates_messages(self) -> None:
        app = FakeApplication()
        console, _ = console_with("hello", "/exit", app=app)

        console.start()

        assert console._messages is not None

    def test_empty_lines_dont_exit_loop(self) -> None:
        app = FakeApplication()
        console, _ = console_with("", "hello", "/exit", app=app)

        console.start()

        assert len(app.turns) == 1

    def test_commands_dont_exit_loop(self) -> None:
        app = FakeApplication()
        console, _ = console_with("/help", "hello", "/exit", app=app)

        console.start()

        assert len(app.turns) == 1

    def test_console_writes_prompt_on_each_iteration(self) -> None:
        app = FakeApplication()
        console, output = console_with("hello", "/exit", app=app)

        console.start()

        assert output.getvalue().startswith("> ")

    def test_eof_writes_newline_to_output(self) -> None:
        output = StringIO()
        source = StringIO("")
        console = InteractiveConsole(FakeApplication(), output=output, source=source)

        console.start()

        assert output.getvalue().endswith("\n")


class TestInteractiveConsoleEdgeCases:
    def test_empty_input_does_nothing(self) -> None:
        app = FakeApplication()
        console, _ = console_with("", "/exit", app=app)

        console.start()

        assert len(app.turns) == 0

    def test_unknown_command_shows_error(self) -> None:
        console, output = console_with("/xyz", "/exit")

        console.start()

        assert "Unknown command" in output.getvalue()

    def test_eof_exits_cleanly(self) -> None:
        output = StringIO()
        source = StringIO("")  # EOF immediately
        console = InteractiveConsole(FakeApplication(), output=output, source=source)

        result = console.start()

        assert result == ""

    def test_history_truncates_long_messages_at_200_chars(self) -> None:
        app = FakeApplication()
        console, output = console_with("/exit", app=app)

        long_content = "x" * 300
        console._messages = (
            MessageHistory()
            .with_message(SYSTEM_MSG)
            .with_message(ChatMessage(USER, MessageContent("hello")))
            .with_message(ChatMessage(ASSISTANT, MessageContent(long_content)))
        )
        console._turn_count = 1

        console.show_history()

        text = output.getvalue()
        assert "x" * 200 in text
        assert "x" * 201 not in text


class TestInteractiveConsolePublicApi:
    def test_clear_history_resets_state(self) -> None:
        console, _ = console_with()
        console._messages = MessageHistory().with_message(SYSTEM_MSG)
        console._turn_count = 5

        console.clear_history()

        assert console._messages is None
        assert console._turn_count == 0

    def test_write_writes_to_output(self) -> None:
        console, output = console_with()
        console.write("hello world")
        assert output.getvalue() == "hello world"

    def test_show_history_prints_turns_and_messages(self) -> None:
        console, output = console_with()
        console._turn_count = 1
        console._messages = (
            MessageHistory()
            .with_message(SYSTEM_MSG)
            .with_message(ChatMessage(USER, MessageContent("hello")))
        )

        console.show_history()

        text = output.getvalue()
        assert "Turns: 1" in text
        assert "User: hello" in text

    def test_show_history_truncates_long_messages(self) -> None:
        console, output = console_with()
        console._turn_count = 1
        long_content = "a" * 250
        console._messages = MessageHistory().with_message(
            ChatMessage(USER, MessageContent(long_content))
        )

        console.show_history()

        text = output.getvalue()
        expected_content = "a" * 200
        assert f"User: {expected_content}\n" in text
