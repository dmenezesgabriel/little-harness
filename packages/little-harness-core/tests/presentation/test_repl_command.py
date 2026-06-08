"""Tests for ReplCommand protocol, CommandRegistry, and built-in commands."""

from __future__ import annotations

import logging
from io import StringIO

import pytest
from little_harness.domain.message import ChatMessage
from little_harness.domain.message_history import MessageHistory
from little_harness.domain.result import AgentResult
from little_harness.domain.steps import AgentSteps
from little_harness.domain.values.numeric_values import ElapsedSeconds
from little_harness.domain.values.role import ASSISTANT, SYSTEM, USER
from little_harness.domain.values.text_values import MessageContent, Prompt
from little_harness.presentation.cli.interactive_console import InteractiveConsole
from little_harness.presentation.cli.repl_command import (
    ClearCommand,
    CommandRegistry,
    ExitCommand,
    ExitReplError,
    HelpCommand,
    HistoryCommand,
    ReplCommand,
    build_default_registry,
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


def make_console() -> InteractiveConsole:
    return InteractiveConsole(FakeApplication(), output=StringIO(), source=StringIO())


# ── ReplCommand protocol structural conformance ─────────────────────────────


class TestReplCommandStructuralConformance:
    def test_clear_command_conforms(self) -> None:
        cmd: ReplCommand = ClearCommand()
        assert isinstance(cmd, ClearCommand)

    def test_exit_command_conforms(self) -> None:
        cmd: ReplCommand = ExitCommand()
        assert isinstance(cmd, ExitCommand)

    def test_help_command_conforms(self) -> None:
        cmd: ReplCommand = HelpCommand()
        assert isinstance(cmd, HelpCommand)

    def test_history_command_conforms(self) -> None:
        cmd: ReplCommand = HistoryCommand()
        assert isinstance(cmd, HistoryCommand)


# ── CommandRegistry ─────────────────────────────────────────────────────────


class TestCommandRegistry:
    def test_get_returns_none_for_unknown(self) -> None:
        registry = CommandRegistry()
        assert registry.get("/nonexistent") is None

    def test_round_trips_a_command(self) -> None:
        registry = CommandRegistry()
        registry.add(ExitCommand(), "built-in")
        assert registry.get("/exit") is not None
        assert registry.get("/quit") is not None

    def test_get_is_case_insensitive(self) -> None:
        registry = CommandRegistry()
        registry.add(ExitCommand(), "built-in")
        assert registry.get("/EXIT") is not None
        assert registry.get("/Quit") is not None

    def test_add_strips_leading_slash_from_alias(self) -> None:
        registry = CommandRegistry()
        registry.add(ExitCommand(), "built-in")
        assert registry.get("exit") is None
        assert registry.get("/exit") is not None

    def test_iter_yields_each_unique_command(self) -> None:
        registry = CommandRegistry()
        registry.add(ClearCommand(), "built-in")
        registry.add(ExitCommand(), "built-in")
        registry.add(HelpCommand(), "built-in")
        registry.add(HistoryCommand(), "built-in")

        names = {c.name for c in registry}
        assert names == {"clear", "exit", "help", "history"}

    def test_iter_yields_no_duplicates_for_aliased_commands(self) -> None:
        registry = CommandRegistry()
        registry.add(ExitCommand(), "built-in")

        items = list(registry)
        assert len(items) == 1
        assert items[0].name == "exit"

    def test_add_default_source_is_built_in(self) -> None:
        registry = CommandRegistry()
        registry.add(ClearCommand(), "built-in")
        assert registry._sources["/clear"] == "built-in"


def test_command_registry_default_source() -> None:
    registry = CommandRegistry()
    registry.add(ClearCommand(), "built-in")
    assert registry._sources["/clear"] == "built-in"


# ── Override tracking ───────────────────────────────────────────────────────


class TestCommandRegistryOverrides:
    def test_override_is_tracked(self) -> None:
        registry = CommandRegistry()

        class CustomClear:
            name = "clear"
            aliases: tuple[str, ...] = ()
            description = "Custom clear"

            def execute(self, _console: object, /) -> None:
                pass

        registry.add(ClearCommand(), "built-in")
        registry.add(CustomClear(), "plugin:custom")

        assert registry.overrides == {"/clear": "built-in \u2192 plugin:custom"}

    def test_overrides_property_returns_copy(self) -> None:
        registry = CommandRegistry()

        class CustomClear:
            name = "clear"
            aliases: tuple[str, ...] = ()
            description = "Custom clear"

            def execute(self, _console: object, /) -> None:
                pass

        registry.add(CustomClear(), "plugin:custom")
        result = registry.overrides
        assert result == {}

    def test_no_overrides_for_unique_commands(self) -> None:
        registry = CommandRegistry()
        registry.add(ClearCommand(), "built-in")
        registry.add(ExitCommand(), "built-in")

        assert registry.overrides == {}

    def test_override_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        registry = CommandRegistry()

        class CustomClear:
            name = "clear"
            aliases: tuple[str, ...] = ()
            description = "Custom clear"

            def execute(self, _console: object, /) -> None:
                pass

        with caplog.at_level(logging.WARNING):
            registry.add(ClearCommand(), "built-in")
            registry.add(CustomClear(), "plugin:custom")

        assert len(caplog.records) == 1
        assert (
            caplog.records[0].message
            == "REPL command '/clear' overridden by plugin:custom (was built-in)"
        )
        assert registry.overrides == {"/clear": "built-in \u2192 plugin:custom"}


class FakeReplConsole:
    def __init__(self, registry: CommandRegistry | None = None) -> None:
        self.history_cleared = False
        self.history_shown = False
        self.written_text = ""
        self._registry = registry or CommandRegistry()

    @property
    def registry(self) -> CommandRegistry:
        return self._registry

    def clear_history(self) -> None:
        self.history_cleared = True

    def show_history(self) -> None:
        self.history_shown = True

    def write(self, text: str) -> None:
        self.written_text += text


# ── ClearCommand ────────────────────────────────────────────────────────────


class TestClearCommand:
    def test_name_and_aliases(self) -> None:
        cmd = ClearCommand()
        assert cmd.name == "clear"
        assert cmd.aliases == ()

    def test_execute_calls_clear_history(self) -> None:
        console = FakeReplConsole()
        ClearCommand().execute(console)
        assert console.history_cleared is True


# ── ExitCommand ─────────────────────────────────────────────────────────────


class TestExitCommand:
    def test_name_and_aliases(self) -> None:
        cmd = ExitCommand()
        assert cmd.name == "exit"
        assert cmd.aliases == ("quit",)

    def test_execute_raises_exit_repl_error(self) -> None:
        console = FakeReplConsole()
        with pytest.raises(ExitReplError):
            ExitCommand().execute(console)


# ── HelpCommand ──────────────────────────────────────────────────────────────


class TestHelpCommand:
    def test_name_and_aliases(self) -> None:
        cmd = HelpCommand()
        assert cmd.name == "help"
        assert cmd.aliases == ()

    def test_execute_prints_available_commands(self) -> None:
        registry = build_default_registry()
        console = FakeReplConsole(registry)

        HelpCommand().execute(console)

        text = console.written_text
        assert "Available commands:" in text
        assert "/exit" in text
        assert "/clear" in text
        assert "/help" in text
        assert "/history" in text

    def test_help_mentions_quit_as_alias(self) -> None:
        registry = build_default_registry()
        console = FakeReplConsole(registry)

        HelpCommand().execute(console)

        text = console.written_text
        assert "/exit" in text
        assert "/quit" in text

    def test_execute_prints_exact_help_format(self) -> None:
        registry = build_default_registry()
        console = FakeReplConsole(registry)

        HelpCommand().execute(console)

        expected = (
            "Available commands:\n"
            "  /clear                   Clear conversation history\n"
            "  /exit, /quit             Exit the interactive session\n"
            "  /help                    Show this help message\n"
            "  /history                 Show conversation history\n"
        )
        assert console.written_text == expected


# ── HistoryCommand ──────────────────────────────────────────────────────────


class TestHistoryCommand:
    def test_name_and_aliases(self) -> None:
        cmd = HistoryCommand()
        assert cmd.name == "history"
        assert cmd.aliases == ()

    def test_execute_calls_show_history(self) -> None:
        console = FakeReplConsole()
        HistoryCommand().execute(console)
        assert console.history_shown is True


# ── build_default_registry ──────────────────────────────────────────────────


class TestBuildDefaultRegistry:
    def test_returns_registry_with_all_builtins(self) -> None:
        registry = build_default_registry()

        assert registry.get("/clear") is not None
        assert registry.get("/exit") is not None
        assert registry.get("/help") is not None
        assert registry.get("/history") is not None

    def test_each_command_is_instantiated(self) -> None:
        registry = build_default_registry()
        names = {c.name for c in registry}

        assert names == {"clear", "exit", "help", "history"}

    def test_registered_as_source_built_in(self) -> None:
        registry = build_default_registry()

        class CustomClear:
            name = "clear"
            aliases: tuple[str, ...] = ()
            description = "Custom clear"

            def execute(self, _console: object, /) -> None:
                pass

        registry.add(CustomClear(), "plugin:custom")

        assert registry.overrides == {"/clear": "built-in \u2192 plugin:custom"}
