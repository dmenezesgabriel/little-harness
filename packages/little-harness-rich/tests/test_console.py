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
from little_harness.presentation.cli.repl_command import (
    ExitReplError,
    build_default_registry,
)
from little_harness_rich.console import RichInteractiveConsole
from rich.console import Console
from rich.panel import Panel


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


class FakeCommandRegistry:
    def __init__(self) -> None:
        self._index: dict[str, object] = {"/test": object()}


class TestRichInteractiveConsole:
    def test_default_console_is_constructed_on_init(self) -> None:
        app = FakeApplication()
        console = RichInteractiveConsole(app)
        assert console._console is not None
        assert isinstance(console._console, Console)

    def test_init_defaults_registry(self) -> None:
        app = FakeApplication()
        # Case 1: registry is None (should fall back to default)
        console = RichInteractiveConsole(app, registry=None)
        assert console._registry is not None
        assert len(console._registry._index) > 0

        # Case 2: registry is provided
        custom_registry = FakeCommandRegistry()
        console2 = RichInteractiveConsole(app, registry=custom_registry)  # type: ignore[arg-type]
        assert console2._registry is custom_registry

    def test_loop_executes_turns_and_commands(  # noqa: PLR0915
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

        prompt_asks: list[str] = []

        def mock_ask(prompt_val: str = "", *_args: object, **_kwargs: object) -> str:
            prompt_asks.append(prompt_val)
            try:
                return next(input_iter)
            except StopIteration as err:
                raise KeyboardInterrupt() from err

        monkeypatch.setattr("rich.prompt.Prompt.ask", mock_ask)

        status_calls: list[object] = []
        original_status = console._console.status

        def mock_status(status_text: object, **kwargs: object) -> object:
            status_calls.append(status_text)
            return original_status(status_text, **kwargs)  # type: ignore[arg-type]

        console._console.status = mock_status  # type: ignore[assignment]

        # Act
        res = console.start()

        # Assert
        assert res == ""
        output_text = output.getvalue()
        assert "Agent Interactive Session" in output_text
        assert "Mocked agent response" in output_text
        assert "Available commands:" in output_text
        assert "/help" in output_text
        assert "/exit" in output_text

        # Verify agent was invoked
        assert len(app.turns) == 1
        assert app.turns[0][0].value == "Hello"

        # Verify correct prompt was displayed
        assert prompt_asks == [">", ">", ">"]

        # Verify correct status was displayed during turn execution
        assert status_calls == ["[bold blue]Agent is thinking...[/bold blue]"]

        # Verify all inputs were consumed (no early breaks)
        with pytest.raises(StopIteration):
            next(input_iter)

        # Verify messages history is updated in the console
        assert console._messages is not None
        assert len(list(console._messages)) == 2

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

        # Assert: exits gracefully and outputs exiting message on the final line
        lines = output.getvalue().splitlines()
        assert lines[-1] == "Exiting..."

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

        # Assert: exits gracefully and outputs exiting message on the final line
        lines = output.getvalue().splitlines()
        assert lines[-1] == "Exiting..."

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
        assert output.getvalue() == "History cleared.\n"

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

    def test_show_history_prints_with_correct_styles(self) -> None:
        app = FakeApplication()
        console = RichInteractiveConsole(app)

        # Spy on console print calls
        printed_args: list[tuple[object, ...]] = []

        def mock_print(*args: object, **_kwargs: object) -> None:
            printed_args.append(args)

        console._console.print = mock_print  # type: ignore[assignment]

        console._turn_count = 1
        console._messages = (
            MessageHistory()
            .with_message(ChatMessage(SYSTEM, MessageContent("hello system")))
            .with_message(ChatMessage(USER, MessageContent("hello user")))
            .with_message(ChatMessage(ASSISTANT, MessageContent("hello assistant")))
        )

        # Act
        console.show_history()

        # Assert correct style tags and formatting are used
        assert printed_args[0] == ("Turns: 1",)
        assert printed_args[1] == ("[bold magenta]System:[/bold magenta]",)
        assert printed_args[4] == ("[bold green]User:[/bold green]",)
        assert printed_args[7] == ("[bold blue]Assistant:[/bold blue]",)

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

        # Verify all inputs were consumed (no early breaks)
        with pytest.raises(StopIteration):
            next(input_iter)

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

        # Verify all inputs were consumed
        with pytest.raises(StopIteration):
            next(input_iter)

    def test_system_messages_cached(self) -> None:
        # Arrange
        app = FakeApplication()
        console = RichInteractiveConsole(app, build_default_registry())

        # Act
        msg1 = console._system_messages()
        msg2 = console._system_messages()

        # Assert
        assert msg1 is msg2
        assert next(iter(msg1)).content.value == "You are a helpful assistant."

    def test_run_turn_increments_turn_count(self) -> None:
        app = FakeApplication()
        console = RichInteractiveConsole(app)
        console._turn_count = 5
        console._run_turn("hello")
        assert console._turn_count == 6

    def test_start_prints_welcome_panel(self) -> None:
        app = FakeApplication()
        console = RichInteractiveConsole(app, build_default_registry())

        panel_printed: list[Panel] = []

        def mock_print(arg: object, **_kwargs: object) -> None:
            if isinstance(arg, Panel):
                panel_printed.append(arg)

        console._console.print = mock_print  # type: ignore[assignment]

        def mock_loop() -> None:
            raise ExitReplError()

        console._loop = mock_loop

        # Act
        console.start()

        # Assert welcome panel attributes
        assert len(panel_printed) == 1
        panel = panel_printed[0]
        assert panel.title == "Agent Interactive Session"
        assert panel.border_style == "cyan"
        assert panel.renderable == (
            "[bold green]Welcome to the Little Harness Agent![/bold green]\n"
            "Type your prompt to start, or [cyan]/help[/cyan] "
            "for available commands."
        )
