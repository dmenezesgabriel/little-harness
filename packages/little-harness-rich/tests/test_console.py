# pyright: reportPrivateUsage=false
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, cast

import pytest
from little_harness.domain.decision import ToolCall
from little_harness.domain.message import ChatMessage
from little_harness.domain.message_history import MessageHistory
from little_harness.domain.result import AgentResult
from little_harness.domain.steps import AgentSteps
from little_harness.domain.values.numeric_values import ElapsedSeconds, Iteration
from little_harness.domain.values.role import ASSISTANT, SYSTEM, USER
from little_harness.domain.values.text_values import (
    MessageContent,
    RunId,
    ToolInput,
    ToolName,
)
from little_harness.domain.values.text_values import (
    Prompt as AgentPrompt,
)
from little_harness.presentation.cli.repl_command import (
    build_default_registry,
)
from little_harness_rich.app import HarnessTuiApp, _TuiObserver, _TuiTokenSink
from little_harness_rich.console import RichInteractiveConsole
from little_harness_rich.permission import RichPermissionRequester
from little_harness_rich.state import ActiveAppState, get_active_app, set_active_app
from little_harness_rich.widgets.chat_input import ChatInputWidget
from little_harness_rich.widgets.chat_message import ChatMessageWidget
from little_harness_rich.widgets.reasoning import ReasoningBlockWidget
from little_harness_rich.widgets.tool_call import ToolCallWidget
from rich.syntax import Syntax
from textual.containers import VerticalScroll
from textual.widgets import Button

if TYPE_CHECKING:
    from little_harness.application.ports.agent_observer import AgentObserver
    from little_harness.domain.decision import AgentDecision
    from little_harness.domain.tool_result import ToolRunResult


class FakeApplication:
    """Fake application that simulates prompt turns and tool calls."""

    def __init__(self) -> None:
        self.turns: list[tuple[AgentPrompt, MessageHistory]] = []
        self.approved: bool | None = None

    def build_system_message(self) -> ChatMessage:
        return ChatMessage(SYSTEM, MessageContent("You are a helpful assistant."))

    def run_turn(
        self, prompt: AgentPrompt, messages: MessageHistory
    ) -> tuple[AgentResult, MessageHistory]:
        self.turns.append((prompt, messages))

        # Test tool permission flow if specified
        if prompt.value == "use_tool":
            requester = RichPermissionRequester()
            call = ToolCall(ToolName("test_tool"), ToolInput('{"arg": 1}'))
            self.approved = requester.request_approval(call)

        # Test tool permission fallback format
        if prompt.value == "use_invalid_tool":
            requester = RichPermissionRequester()
            # Bad JSON syntax causes fallback syntax rendering
            call = ToolCall(ToolName("test_tool"), ToolInput("invalid_json_data"))
            self.approved = requester.request_approval(call)

        result = AgentResult(
            MessageContent("Mocked agent response"),
            ElapsedSeconds(0.5),
            AgentSteps(),
        )
        updated = messages.with_message(
            ChatMessage(ASSISTANT, MessageContent("Mocked agent response"))
        )
        return result, updated


class TestChatMessageWidget:
    """Test suite for ChatMessageWidget."""

    @pytest.mark.asyncio
    async def test_factory_methods(self) -> None:
        app = HarnessTuiApp(FakeApplication(), build_default_registry())
        async with app.run_test():
            user_msg = ChatMessageWidget.user("hello")
            assert user_msg.role == "user"
            assert user_msg.text_content == "hello"

            assistant_msg = ChatMessageWidget.assistant("response")
            assert assistant_msg.role == "assistant"
            assert assistant_msg.text_content == "response"

            system_msg = ChatMessageWidget.system("system log")
            assert system_msg.role == "system"
            assert system_msg.text_content == "system log"

    @pytest.mark.asyncio
    async def test_message_mounting_and_classes(self) -> None:
        app = HarnessTuiApp(FakeApplication(), build_default_registry())
        async with app.run_test() as pilot:
            user_msg = ChatMessageWidget.user("hello")
            assistant_msg = ChatMessageWidget.assistant("response")
            system_msg = ChatMessageWidget.system("system log")

            # Mount to execute on_mount and verify styles
            stream = app.query_one("#message-stream")
            await stream.mount(user_msg)
            await stream.mount(assistant_msg)
            await stream.mount(system_msg)
            await pilot.pause()

            assert user_msg.has_class("chat-bubble")
            assert user_msg.has_class("user")
            assert assistant_msg.has_class("chat-bubble")
            assert assistant_msg.has_class("assistant")
            assert system_msg.has_class("chat-bubble")
            assert system_msg.has_class("system")

            # Test empty update path
            user_msg.update_content("")
            assert user_msg.text_content == ""
            assert user_msg._Static__content == ""  # type: ignore


class TestChatInputWidget:
    """Test suite for ChatInputWidget."""

    @pytest.mark.asyncio
    async def test_value_getter_setter(self) -> None:
        app = HarnessTuiApp(FakeApplication(), build_default_registry())
        async with app.run_test():
            chat_input = app.query_one(ChatInputWidget)
            assert chat_input.input.placeholder == "Type your message or /help..."
            chat_input.value = "test value"
            assert chat_input.value == "test value"
            assert chat_input.input.value == "test value"
            chat_input.focus()

            # Default constructor placeholder test
            default_widget = ChatInputWidget()
            assert default_widget.input.placeholder == "Type your prompt..."


class TestReasoningBlockWidget:
    """Test suite for ReasoningBlockWidget."""

    @pytest.mark.asyncio
    async def test_update_and_complete(self) -> None:
        app = HarnessTuiApp(FakeApplication(), build_default_registry())
        async with app.run_test() as pilot:
            widget = ReasoningBlockWidget()
            assert widget._collapsible is None  # type: ignore

            await app.query_one("#message-stream").mount(widget)
            await pilot.pause()
            assert "Thinking..." in widget.title
            assert widget.collapsed is False
            assert widget._collapsible is not None  # type: ignore
            assert widget._text_widget.parent is not None

            widget.update_reasoning("Analyzing files...")
            assert widget._text_widget._Static__content == "Analyzing files..."  # type: ignore

            widget.complete()
            assert widget.title == "Thought Process (Done)"
            assert widget.collapsed is True

    @pytest.mark.asyncio
    async def test_append_reasoning(self) -> None:
        app = HarnessTuiApp(FakeApplication(), build_default_registry())
        async with app.run_test() as pilot:
            widget = ReasoningBlockWidget()
            await app.query_one("#message-stream").mount(widget)
            await pilot.pause()

            widget.append_reasoning("Thinking step 1...")
            assert widget._accumulated_text == "Thinking step 1..."
            widget.append_reasoning(" step 2.")
            assert widget._accumulated_text == "Thinking step 1... step 2."


class TestToolCallWidget:
    """Test suite for ToolCallWidget."""

    @pytest.mark.asyncio
    async def test_render_tool_details_json(self) -> None:
        app = HarnessTuiApp(FakeApplication(), build_default_registry())
        async with app.run_test() as pilot:
            call = ToolCall(ToolName("json_tool"), ToolInput('{"key": "value"}'))
            widget = ToolCallWidget(call)
            await app.query_one("#message-stream").mount(widget)
            await pilot.pause()
            syntax_obj = cast(Syntax, widget._details._Static__content)  # type: ignore
            assert widget.tool_call.tool_name.value == "json_tool"
            assert isinstance(syntax_obj, Syntax)
            assert syntax_obj.lexer is not None
            assert syntax_obj.lexer.name == "JSON"
            assert widget._status.id == "status-container"

    @pytest.mark.asyncio
    async def test_render_tool_details_raw_text(self) -> None:
        app = HarnessTuiApp(FakeApplication(), build_default_registry())
        async with app.run_test() as pilot:
            call = ToolCall(ToolName("raw_tool"), ToolInput("plain text"))
            widget = ToolCallWidget(call)
            await app.query_one("#message-stream").mount(widget)
            await pilot.pause()
            syntax_obj = cast(Syntax, widget._details._Static__content)  # type: ignore
            assert widget.tool_call.tool_name.value == "raw_tool"
            assert isinstance(syntax_obj, Syntax)
            assert syntax_obj.lexer is not None
            assert syntax_obj.lexer.name == "Text only"

    @pytest.mark.asyncio
    async def test_tool_call_widget_with_future(self) -> None:
        app = HarnessTuiApp(FakeApplication(), build_default_registry())
        async with app.run_test() as pilot:
            future: asyncio.Future[bool] = asyncio.Future()
            call = ToolCall(ToolName("test_tool"), ToolInput('{"key": "value"}'))
            widget = ToolCallWidget(call, future)
            await app.query_one("#message-stream").mount(widget)
            await pilot.pause()

            assert widget.has_class("tool-call-card")
            assert widget._status.id == "status-container"

            buttons_container = widget.query_one("#buttons-container")
            assert buttons_container.id == "buttons-container"

            approve_btn = buttons_container.query_one("#approve", Button)
            reject_btn = buttons_container.query_one("#reject", Button)
            assert approve_btn.label == "Approve"
            assert reject_btn.label == "Reject"


class TestTuiAppAndConsole:
    """Test suite for HarnessTuiApp and RichInteractiveConsole adapter."""

    @pytest.mark.asyncio
    async def test_app_lifecycle_and_input(self) -> None:
        app_core = FakeApplication()
        app = HarnessTuiApp(app_core, build_default_registry())

        async with app.run_test() as pilot:
            # Mount welcome panels
            stream = app.query_one("#message-stream", VerticalScroll)
            assert len(stream.children) == 2

            # Input normal text
            chat_input = app.query_one(ChatInputWidget)
            chat_input.value = "Hello"
            await pilot.press("enter")

            # Check that user prompt bubble got added
            # Note: since the run turn can finish instantly,
            # we expect 4 children after pause (thinking is disabled by default)
            await pilot.pause(0.8)
            assert len(stream.children) == 4

            user_bubble = stream.children[2]
            assert isinstance(user_bubble, ChatMessageWidget)
            assert user_bubble.role == "user"
            assert user_bubble.text_content == "Hello"

            # Check that agent responded
            assistant_bubble = stream.children[3]
            assert isinstance(assistant_bubble, ChatMessageWidget)
            assert assistant_bubble.role == "assistant"
            assert assistant_bubble.text_content == "Mocked agent response"
            assert len(app_core.turns) == 1

    @pytest.mark.asyncio
    async def test_app_slash_command_unknown(self) -> None:
        app_core = FakeApplication()
        app = HarnessTuiApp(app_core, build_default_registry())

        async with app.run_test() as pilot:
            stream = app.query_one("#message-stream", VerticalScroll)

            # Test command autocomplete / unknown
            chat_input = app.query_one(ChatInputWidget)
            chat_input.value = "/unknown"
            await pilot.press("enter")
            await pilot.pause()

            # The system bubble with warning is added
            last_child = stream.children[-1]
            assert isinstance(last_child, ChatMessageWidget)
            assert "Unknown command" in last_child.text_content

    @pytest.mark.asyncio
    async def test_app_slash_command_clear(self) -> None:
        app_core = FakeApplication()
        app = HarnessTuiApp(app_core, build_default_registry())

        async with app.run_test() as pilot:
            stream = app.query_one("#message-stream", VerticalScroll)
            chat_input = app.query_one(ChatInputWidget)

            # Test /clear command
            app._turn_count = 3
            chat_input.value = "/clear"
            await pilot.press("enter")
            await pilot.pause()

            assert app._turn_count == 0
            assert app._messages is None
            assert len(stream.children) == 1
            first_child = stream.children[0]
            assert isinstance(first_child, ChatMessageWidget)
            assert "History cleared" in first_child.text_content

    @pytest.mark.asyncio
    async def test_app_show_history_command(self) -> None:
        app_core = FakeApplication()
        app = HarnessTuiApp(app_core, build_default_registry())

        async with app.run_test() as pilot:
            stream = app.query_one("#message-stream", VerticalScroll)

            # Mock messages history
            app._messages = (
                MessageHistory()
                .with_message(ChatMessage(USER, MessageContent("hello u")))
                .with_message(
                    ChatMessage(ASSISTANT, MessageContent("hello a")),
                )
            )
            app._turn_count = 1

            chat_input = app.query_one(ChatInputWidget)
            chat_input.value = "/history"
            await pilot.press("enter")
            await pilot.pause()

            # Ensure history was written to stream
            history_bubble = stream.children[-2]
            bubble = cast(ChatMessageWidget, history_bubble)
            assert bubble.role == "user"
            assert bubble.text_content == "hello u"

    @pytest.mark.asyncio
    async def test_tool_call_approval_dialog_yes(self) -> None:
        app_core = FakeApplication()
        app = HarnessTuiApp(app_core, build_default_registry())

        async with app.run_test() as pilot:
            chat_input = app.query_one(ChatInputWidget)
            chat_input.value = "use_tool"
            await pilot.press("enter")

            # Pause to let the thread spawn and request approval
            await pilot.pause(0.2)

            # A tool call widget should be mounted
            tool_widget = app.query_one(ToolCallWidget)
            assert tool_widget is not None

            # Simulate pressing the "Approve" button
            approve_btn = tool_widget.query_one("#approve", Button)
            approve_btn.press()
            await pilot.pause(0.8)

            assert app_core.approved is True
            # Verify tool status text shows Approved
            content = cast(object, tool_widget._status._Static__content)  # type: ignore
            assert "Approved" in str(content)

    @pytest.mark.asyncio
    async def test_tool_call_approval_dialog_no(self) -> None:
        app_core = FakeApplication()
        app = HarnessTuiApp(app_core, build_default_registry())

        async with app.run_test() as pilot:
            chat_input = app.query_one(ChatInputWidget)
            chat_input.value = "use_invalid_tool"
            await pilot.press("enter")

            # Let the prompt render
            await pilot.pause(0.2)

            tool_widget = app.query_one(ToolCallWidget)

            # Simulate pressing the "Reject" button
            reject_btn = tool_widget.query_one("#reject", Button)
            reject_btn.press()
            await pilot.pause(0.8)

            assert app_core.approved is False
            content = cast(object, tool_widget._status._Static__content)  # type: ignore
            assert "Rejected" in str(content)


class TestConsoleAdapter:
    """Test console wrapper matching the InteractiveRunner interface."""

    def test_active_app_state_initial_value(self) -> None:
        state = ActiveAppState()
        assert state.app is None

    def test_console_forwarders(self, monkeypatch: pytest.MonkeyPatch) -> None:
        app_core = FakeApplication()
        console = RichInteractiveConsole(app_core)
        assert console.registry is not None

        # When active app is None, forwarders should not raise errors
        assert get_active_app() is None
        console.clear_history()
        console.show_history()
        console.write("test")

        # Mock app active state
        class DummyApp:
            def __init__(self) -> None:
                self.calls: list[str] = []

            def clear_history(self) -> None:
                self.calls.append("clear")

            def show_history(self) -> None:
                self.calls.append("history")

            def write(self, text: str) -> None:
                self.calls.append(f"write:{text}")

        dummy = DummyApp()
        set_active_app(dummy)  # type: ignore[arg-type]

        try:
            console.clear_history()
            console.show_history()
            console.write("announce")
            assert dummy.calls == ["clear", "history", "write:announce"]
        finally:
            set_active_app(None)

    def test_console_start_runs_app(self, monkeypatch: pytest.MonkeyPatch) -> None:
        app_core = FakeApplication()
        console = RichInteractiveConsole(app_core)
        run_called = False

        def mock_run(_self_app: object) -> None:
            nonlocal run_called
            run_called = True

        monkeypatch.setattr(HarnessTuiApp, "run", mock_run)

        res = console.start()
        assert res == ""
        assert run_called is True
        assert get_active_app() is None


class FakeTokenSink:
    def __init__(self) -> None:
        self.chunks: list[object] = []

    def emit(self, chunk: object) -> None:
        self.chunks.append(chunk)


class FakeDependencies:
    def __init__(self, observer: AgentObserver, token_sink: object) -> None:
        self.observer = observer
        self.token_sink = token_sink


class FakeRuntime:
    def __init__(self, observer: AgentObserver, token_sink: object) -> None:
        self._dependencies = FakeDependencies(observer, token_sink)


class FakeApplicationWithRuntime:
    def __init__(
        self, observer: AgentObserver, token_sink: object | None = None
    ) -> None:
        self.token_sink = token_sink or FakeTokenSink()
        self._runtime = FakeRuntime(observer, self.token_sink)
        self.turns: list[tuple[AgentPrompt, MessageHistory]] = []

    def build_system_message(self) -> ChatMessage:
        return ChatMessage(SYSTEM, MessageContent("System message"))

    def run_turn(
        self, prompt: AgentPrompt, messages: MessageHistory
    ) -> tuple[AgentResult, MessageHistory]:
        self.turns.append((prompt, messages))
        run_id = RunId("test-run")
        iteration = Iteration(1)

        # Fire observer events to trigger TuiObserver handlers
        self._runtime._dependencies.observer.on_run_started(run_id, prompt)
        self._runtime._dependencies.observer.on_model_completed(
            run_id,
            iteration,
            MessageContent('{"thought": "Step-by-step thinking"}'),
            ElapsedSeconds(0.1),
        )
        self._runtime._dependencies.observer.on_model_completed(
            run_id,
            iteration,
            MessageContent("Plain text fallback"),
            ElapsedSeconds(0.1),
        )
        self._runtime._dependencies.observer.on_decision_parsed(
            run_id, iteration, cast("AgentDecision", None)
        )
        self._runtime._dependencies.observer.on_repair(
            run_id, iteration, ValueError("invalid JSON syntax")
        )

        class DummyToolName:
            value = "calculator"

        class DummyToolRunResult:
            tool_name = DummyToolName()
            succeeded = True

        self._runtime._dependencies.observer.on_tool_invoked(
            run_id,
            iteration,
            cast("ToolRunResult", DummyToolRunResult()),
            ElapsedSeconds(0.1),
        )
        self._runtime._dependencies.observer.on_run_finished(
            run_id, cast("AgentResult", None)
        )

        result = AgentResult(
            MessageContent("Final result"),
            ElapsedSeconds(0.5),
            AgentSteps(),
        )
        updated = messages.with_message(
            ChatMessage(ASSISTANT, MessageContent("Final result"))
        )
        return result, updated


class OriginalObserver:
    def __init__(self) -> None:
        self.events: list[str] = []

    def on_run_started(self, run_id: object, prompt: object) -> None:
        self.events.append("started")

    def on_model_completed(
        self,
        run_id: object,
        iteration: object,
        output: object,
        elapsed: object,
    ) -> None:
        self.events.append("completed")

    def on_decision_parsed(
        self, run_id: object, iteration: object, decision: object
    ) -> None:
        self.events.append("parsed")

    def on_repair(self, run_id: object, iteration: object, error: object) -> None:
        self.events.append("repair")

    def on_tool_invoked(
        self,
        run_id: object,
        iteration: object,
        result: object,
        elapsed: object,
    ) -> None:
        self.events.append("tool")

    def on_run_finished(self, run_id: object, result: object) -> None:
        self.events.append("finished")


class TestTuiObserver:
    """Test suite for TuiObserver and app event integration."""

    @pytest.mark.asyncio
    async def test_tui_observer_integration(self) -> None:
        orig_obs = OriginalObserver()
        app_core = FakeApplicationWithRuntime(orig_obs)
        app = HarnessTuiApp(app_core, build_default_registry())

        # Check observer is wrapped
        assert isinstance(app_core._runtime._dependencies.observer, _TuiObserver)

        async with app.run_test() as pilot:
            chat_input = app.query_one(ChatInputWidget)
            chat_input.value = "run task"
            await pilot.press("enter")
            # Wait for background thread worker turn to run and call observer methods
            await pilot.pause(0.5)

            # Check original observer was called
            assert orig_obs.events == [
                "started",
                "completed",
                "completed",
                "parsed",
                "repair",
                "tool",
                "finished",
            ]

            # Verify widgets in TUI stream
            stream = app.query_one("#message-stream", VerticalScroll)

            # Check repair and tool logs were written to the stream
            texts = [
                child.text_content
                for child in stream.children
                if isinstance(child, ChatMessageWidget)
            ]
            assert any("Repairing" in t for t in texts)
            assert any("completed" in t for t in texts)

    @pytest.mark.asyncio
    async def test_tui_token_sink_integration(self) -> None:
        orig_obs = OriginalObserver()
        app_core = FakeApplicationWithRuntime(orig_obs)
        app = HarnessTuiApp(app_core, build_default_registry())

        # Check token sink is wrapped
        assert isinstance(app_core._runtime._dependencies.token_sink, _TuiTokenSink)

        async with app.run_test() as pilot:
            # Test token sink emit updates reasoning widget
            reasoning = ReasoningBlockWidget()
            await app.query_one("#message-stream").mount(reasoning)
            app._active_reasoning_widget = reasoning

            app_core._runtime._dependencies.token_sink.emit(
                MessageContent("Token stream content")
            )
            await pilot.pause()
            assert reasoning._accumulated_text == "Token stream content"


class TestAppActionClearHistory:
    """Test clear history keyboard shortcut."""

    @pytest.mark.asyncio
    async def test_app_action_clear_history(self) -> None:
        app_core = FakeApplication()
        app = HarnessTuiApp(app_core, build_default_registry())
        async with app.run_test() as pilot:
            app._turn_count = 5
            await pilot.press("ctrl+l")
            await pilot.pause()
            assert app._turn_count == 0


class DummySchema:
    def __init__(self, props: dict[str, object]) -> None:
        self.value = {"properties": props}


class DummyPolicy:
    def __init__(self, schema_val: DummySchema | None) -> None:
        self._schema_val = schema_val

    def response_schema(self, _specs: object) -> object:
        return self._schema_val


class DummyToolRegistry:
    def specs(self) -> list[object]:
        return []


class DummyDependencies:
    def __init__(self, policy: object) -> None:
        self.policy = policy
        self.tool_registry = DummyToolRegistry()


class DummyRuntime:
    def __init__(self, policy: object) -> None:
        self._dependencies = DummyDependencies(policy)

    @property
    def _runtime(self) -> object:
        return self


class DummyApp(FakeApplication):
    def __init__(self, policy: object) -> None:
        super().__init__()
        self._runtime = DummyRuntime(policy)


class TestTuiThinkingEnabled:
    """Test suite for the conditional thinking/reasoning display in the TUI."""

    def test_thinking_disabled_by_default(self) -> None:
        app = HarnessTuiApp(FakeApplication(), build_default_registry())
        assert app._is_thinking_enabled() is False

    def test_thinking_enabled_with_schema(self) -> None:
        # 1. Schema with 'thought' property -> True
        schema = DummySchema({"thought": {"type": "string"}})
        app = HarnessTuiApp(DummyApp(DummyPolicy(schema)), build_default_registry())
        assert app._is_thinking_enabled() is True

        # 2. Schema without 'thought' property -> False
        schema_no_thought = DummySchema({"other": {"type": "string"}})
        app = HarnessTuiApp(
            DummyApp(DummyPolicy(schema_no_thought)),
            build_default_registry(),
        )
        assert app._is_thinking_enabled() is False

        # 3. None schema -> False
        app = HarnessTuiApp(DummyApp(DummyPolicy(None)), build_default_registry())
        assert app._is_thinking_enabled() is False
