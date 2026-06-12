"""Main Textual TUI Application for little-harness."""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import TYPE_CHECKING, Any, ClassVar, cast

from little_harness.domain.message_history import MessageHistory
from little_harness.domain.values.text_values import Prompt
from little_harness.presentation.cli.repl_command import ExitReplError
from textual._context import active_app
from textual.app import App, ComposeResult
from textual.binding import BindingType
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Footer, Header, Input

from little_harness_rich.state import set_active_app
from little_harness_rich.widgets.chat_input import ChatInputWidget
from little_harness_rich.widgets.chat_message import ChatMessageWidget
from little_harness_rich.widgets.reasoning import ReasoningBlockWidget
from little_harness_rich.widgets.tool_call import ToolCallWidget

if TYPE_CHECKING:
    from little_harness.application.ports.token_sink import TokenSink
    from little_harness.domain.decision import AgentDecision, ToolCall
    from little_harness.domain.result import AgentResult
    from little_harness.domain.tool_result import ToolRunResult
    from little_harness.domain.values.numeric_values import ElapsedSeconds, Iteration
    from little_harness.domain.values.text_values import MessageContent, RunId
    from little_harness.presentation.cli.interactive_console import Application
    from little_harness.presentation.cli.repl_command import CommandRegistry


class HarnessTuiApp(App[str]):
    """A Textual Terminal User Interface for running agent sessions."""

    CSS_PATH = "tcss/default.tcss"

    BINDINGS: ClassVar[list[BindingType]] = [
        ("ctrl+c", "quit", "Exit"),
        ("ctrl+l", "clear_history", "Clear"),
    ]

    def __init__(
        self,
        application: Application,
        registry: CommandRegistry,
    ) -> None:
        """Initialize the TUI application.

        Args:
            application: The agent core runner interface.
            registry: The slash command registry.

        """
        super().__init__()
        self._app = application
        self._command_registry = registry
        self._messages: MessageHistory | None = None
        self._turn_count = 0
        self._active_future: asyncio.Future[bool] | None = None
        self._active_reasoning_widget: ReasoningBlockWidget | None = None
        self._wrap_observer()

    def _wrap_observer(self) -> None:
        try:
            runtime = getattr(self._app, "_runtime", None)
            if runtime is not None:
                dependencies = getattr(runtime, "_dependencies", None)
                if dependencies is not None:
                    original = dependencies.observer
                    dependencies.observer = _TuiObserver(self, original)
                    original_sink = dependencies.token_sink
                    dependencies.token_sink = _TuiTokenSink(self, original_sink)
        except Exception:  # nosec
            pass

    def compose(self) -> ComposeResult:
        """Compose the main TUI layout."""
        yield Header(show_clock=True)
        yield VerticalScroll(id="message-stream")
        with Vertical(id="input-container"):
            yield ChatInputWidget(placeholder="Type your message or /help...")
        yield Footer()

    def on_mount(self) -> None:
        """Initialize and mount the system message bubble."""
        set_active_app(self)
        stream = self.query_one("#message-stream", VerticalScroll)
        stream.mount(ChatMessageWidget.system("Welcome to the Little Harness Agent!"))
        stream.mount(ChatMessageWidget.system("Type your prompt or /help."))
        self.query_one(ChatInputWidget).focus()

    def on_unmount(self) -> None:
        """Clear active app on unmount."""
        set_active_app(None)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle chat input submissions.

        Args:
            event: The input submission event.

        """
        text = event.value.strip()
        if not text:
            return

        chat_input = self.query_one(ChatInputWidget)
        chat_input.value = ""

        # 1. Process slash commands
        if text.startswith("/"):
            await self._process_command(text)
            return

        # 2. Append User message bubble to the TUI
        stream = self.query_one("#message-stream", VerticalScroll)
        await stream.mount(ChatMessageWidget.user(text))
        stream.scroll_end()

        # 3. Spawn background worker for the agent turn
        self.run_worker(self._run_agent_turn(text))

    async def _process_command(self, line: str) -> bool:
        command = self._command_registry.get(line)
        if command is None:
            self.write(f"Unknown command: {line}. Try /help.")
            return True

        try:
            command.execute(self)
        except ExitReplError:
            self.exit("")
        return True

    def clear_history(self) -> None:
        """Clear conversation history, reset state, and update the UI."""
        self._messages = None
        self._turn_count = 0
        stream = self.query_one("#message-stream", VerticalScroll)
        for child in list(stream.children):
            child.remove()
        stream.mount(ChatMessageWidget.system("History cleared."))

    def show_history(self) -> None:
        """Display conversation history in the TUI stream."""
        stream = self.query_one("#message-stream", VerticalScroll)
        stream.mount(ChatMessageWidget.system(f"Turns: {self._turn_count}"))
        if self._messages is None:
            return
        for message in self._messages:
            role = message.role.name.lower()
            stream.mount(ChatMessageWidget(role, message.content.value))

    def write(self, text: str) -> None:
        """Write a system announcement to the stream."""
        stream = self.query_one("#message-stream", VerticalScroll)
        stream.mount(ChatMessageWidget.system(text.strip()))

    @property
    def registry(self) -> CommandRegistry:
        """Get the command registry."""
        return self._command_registry

    def prompt_permission(self, call: ToolCall) -> bool:
        """Prompt the user for tool execution permission (Thread-safe).

        Args:
            call: The tool call details.

        Returns:
            True if approved, False otherwise.

        """
        loop = self._loop
        if loop is None:
            raise RuntimeError("Event loop is not running")
        future = asyncio.run_coroutine_threadsafe(
            self._prompt_permission_coro(call),
            loop,
        )
        return future.result()

    async def _prompt_permission_coro(self, call: ToolCall) -> bool:
        active_app.set(self)
        self._active_future = asyncio.Future()
        stream = self.query_one("#message-stream", VerticalScroll)
        widget = ToolCallWidget(call, self._active_future)
        await stream.mount(widget)
        stream.scroll_end()

        # Disable main chat input during prompt
        chat_input = self.query_one(ChatInputWidget)
        chat_input.input.disabled = True

        # Shift focus to the approve button for keyboard interaction
        with contextlib.suppress(Exception):
            widget.query_one("#approve").focus()

        try:
            approved = await self._active_future
        finally:
            chat_input.input.disabled = False
            chat_input.focus()
            self._active_future = None

        return approved

    def _get_or_build_history(self) -> MessageHistory:
        history = self._messages
        if history is not None:
            return history
        history = MessageHistory().with_message(
            self._app.build_system_message(),
        )
        self._messages = history
        return history

    async def _run_agent_turn(self, text: str) -> None:
        active_app.set(self)
        history = self._get_or_build_history()

        # Start thinking spinner if thinking is enabled in the model policy
        widget = None
        if self._is_thinking_enabled():
            widget = await self._start_thinking()
            self._active_reasoning_widget = widget

        try:
            result, updated = await asyncio.to_thread(
                self._app.run_turn, Prompt(text), history
            )
            await self._render_result(result, updated)
        except Exception as e:
            await self._handle_error(e)
        finally:
            if widget is not None:
                self._stop_thinking(widget)
            self._active_reasoning_widget = None

    def _get_policy_schema(self) -> object | None:
        """Safely retrieve the response schema from the policy."""
        try:
            runtime = cast(Any, self._app)._runtime
            dependencies = runtime._dependencies
            policy = dependencies.policy
            return policy.response_schema(dependencies.tool_registry.specs())
        except Exception:  # nosec
            return None

    def _is_thinking_enabled(self) -> bool:
        """Check if thinking/reasoning protocol is enabled on the model policy."""
        schema = self._get_policy_schema()
        if schema is None:
            return False
        properties = getattr(schema, "value", {}).get("properties", {})
        return "thought" in properties

    async def _start_thinking(self) -> ReasoningBlockWidget:
        stream = self.query_one("#message-stream", VerticalScroll)
        widget = ReasoningBlockWidget()
        await stream.mount(widget)
        stream.scroll_end()
        return widget

    def _stop_thinking(self, widget: ReasoningBlockWidget) -> None:
        widget.complete()

    async def _render_result(
        self,
        result: AgentResult,
        updated: MessageHistory,
    ) -> None:
        self._messages = updated
        self._turn_count += 1
        stream = self.query_one("#message-stream", VerticalScroll)
        await stream.mount(ChatMessageWidget.assistant(result.answer.value))
        stream.scroll_end()

    async def _handle_error(self, exc: Exception) -> None:
        self.write(f"[red]Error during agent turn: {exc}[/red]")

    def action_clear_history(self) -> None:
        """Clear history action handler."""
        self.clear_history()

    def handle_run_started(self, _run_id: RunId, _prompt: Prompt) -> None:
        """Handle run started event."""
        # Active reasoning widget is mounted at turn start

    def handle_model_completed(
        self,
        _run_id: RunId,
        _iteration: Iteration,
        output: MessageContent,
        _elapsed: ElapsedSeconds,
    ) -> None:
        """Handle model completion, extracting and showing reasoning if available."""
        if self._active_reasoning_widget is None:
            return
        try:
            data = json.loads(output.value)
            thought = data.get("thought", "")
            if thought:
                self._active_reasoning_widget.update_reasoning(thought)
                return
            self._active_reasoning_widget.update_reasoning(output.value)
        except Exception:
            self._active_reasoning_widget.update_reasoning(output.value)

    def handle_decision_parsed(
        self,
        _run_id: RunId,
        _iteration: Iteration,
        _decision: AgentDecision,
    ) -> None:
        """Handle parsed decision event."""
        # No specific action needed

    def handle_tool_invoked(
        self,
        _run_id: RunId,
        _iteration: Iteration,
        result: ToolRunResult,
        _elapsed: ElapsedSeconds,
    ) -> None:
        """Handle tool invocation finish, outputting status update."""
        status = "succeeded" if result.succeeded else "failed"
        self.write(f"Tool {result.tool_name.value} completed ({status}).")

    def handle_repair(
        self,
        _run_id: RunId,
        _iteration: Iteration,
        error: Exception,
    ) -> None:
        """Handle model output repair attempt."""
        self.write(f"Repairing invalid model output: {error}")

    def handle_run_finished(self, _run_id: RunId, _result: AgentResult) -> None:
        """Handle run finished event."""
        if self._active_reasoning_widget is not None:
            self._stop_thinking(self._active_reasoning_widget)
            self._active_reasoning_widget = None

    def handle_token_emitted(self, chunk: str) -> None:
        """Handle streamed token chunk by appending to the reasoning block."""
        if self._active_reasoning_widget is not None:
            self._active_reasoning_widget.append_reasoning(chunk)


class _TuiObserver:
    """Observer wrapper that forwards agent events to the TUI app."""

    def __init__(self, app: HarnessTuiApp, delegate: object) -> None:
        """Initialize the observer wrapper."""
        self._app = app
        self._delegate = delegate

    def on_run_started(self, run_id: RunId, prompt: Prompt) -> None:
        """Forward run start event."""
        self._app.call_from_thread(self._app.handle_run_started, run_id, prompt)
        if hasattr(self._delegate, "on_run_started"):
            self._delegate.on_run_started(run_id, prompt)  # type: ignore

    def on_model_completed(
        self,
        run_id: RunId,
        iteration: Iteration,
        output: MessageContent,
        elapsed: ElapsedSeconds,
    ) -> None:
        """Forward model completion event."""
        self._app.call_from_thread(
            self._app.handle_model_completed, run_id, iteration, output, elapsed
        )
        if hasattr(self._delegate, "on_model_completed"):
            self._delegate.on_model_completed(  # type: ignore
                run_id, iteration, output, elapsed
            )

    def on_decision_parsed(
        self, run_id: RunId, iteration: Iteration, decision: AgentDecision
    ) -> None:
        """Forward decision parsed event."""
        self._app.call_from_thread(
            self._app.handle_decision_parsed, run_id, iteration, decision
        )
        if hasattr(self._delegate, "on_decision_parsed"):
            self._delegate.on_decision_parsed(run_id, iteration, decision)  # type: ignore

    def on_tool_invoked(
        self,
        run_id: RunId,
        iteration: Iteration,
        result: ToolRunResult,
        elapsed: ElapsedSeconds,
    ) -> None:
        """Forward tool invoked event."""
        self._app.call_from_thread(
            self._app.handle_tool_invoked, run_id, iteration, result, elapsed
        )
        if hasattr(self._delegate, "on_tool_invoked"):
            self._delegate.on_tool_invoked(  # type: ignore
                run_id, iteration, result, elapsed
            )

    def on_repair(self, run_id: RunId, iteration: Iteration, error: Exception) -> None:
        """Forward model output repair event."""
        self._app.call_from_thread(self._app.handle_repair, run_id, iteration, error)
        if hasattr(self._delegate, "on_repair"):
            self._delegate.on_repair(run_id, iteration, error)  # type: ignore

    def on_run_finished(self, run_id: RunId, result: AgentResult) -> None:
        """Forward run finish event."""
        self._app.call_from_thread(self._app.handle_run_finished, run_id, result)
        if hasattr(self._delegate, "on_run_finished"):
            self._delegate.on_run_finished(run_id, result)  # type: ignore


class _TuiTokenSink:
    """Wrapper that intercepts TokenSink calls and marshals them onto the TUI thread."""

    def __init__(self, app: HarnessTuiApp, delegate: TokenSink) -> None:
        """Initialize the token sink wrapper."""
        self._app = app
        self._delegate = delegate

    def emit(self, chunk: MessageContent) -> None:
        """Forward token chunks and post to TUI thread."""
        try:
            self._app.call_from_thread(self._app.handle_token_emitted, chunk.value)
        except RuntimeError:
            self._app.handle_token_emitted(chunk.value)
        self._delegate.emit(chunk)
