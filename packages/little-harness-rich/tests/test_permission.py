# pyright: reportPrivateUsage=false
from __future__ import annotations

import pytest
from little_harness.domain.decision import ToolCall
from little_harness.domain.values.text_values import ToolInput, ToolName
from little_harness_rich.permission import RichPermissionRequester
from little_harness_rich.state import get_active_app, set_active_app
from rich.console import Console


class FakeApp:
    """Fake application that responds to prompt permissions."""

    def __init__(self) -> None:
        self.called_call: ToolCall | None = None
        self.return_value = True

    def prompt_permission(self, call: ToolCall) -> bool:
        """Prompt user for permission."""
        self.called_call = call
        return self.return_value


class TestRichPermissionRequester:
    def test_request_approval_routes_to_active_app_when_available(self) -> None:
        """Verify request_approval uses TUI app when set_active_app is used."""
        app = FakeApp()
        set_active_app(app)  # type: ignore[arg-type]

        try:
            requester = RichPermissionRequester()
            call = ToolCall(ToolName("bash"), ToolInput('echo "hello"'))

            assert requester.request_approval(call) is True
            assert app.called_call is call

            app.return_value = False
            assert requester.request_approval(call) is False
        finally:
            set_active_app(None)

    def test_request_approval_fallback_to_confirm_ask_returns_true(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify fallback behavior using Confirm.ask when no active app exists."""
        call_args = {}

        def mock_ask(*args: object, **kwargs: object) -> bool:
            call_args["args"] = args
            call_args["kwargs"] = kwargs
            return True

        monkeypatch.setattr("rich.prompt.Confirm.ask", mock_ask)
        console = Console(force_terminal=True)
        requester = RichPermissionRequester(console)

        assert requester._console is console
        assert get_active_app() is None

        call = ToolCall(ToolName("bash"), ToolInput('echo "hello"'))

        assert requester.request_approval(call) is True

        expected_prompt = (
            "Allow tool [cyan]'bash'[/cyan] to run with input "
            "[yellow]'echo \"hello\"'[/yellow]?"
        )
        assert call_args["args"] == (expected_prompt,)
        assert call_args["kwargs"] == {"console": console, "default": False}

    def test_request_approval_fallback_to_confirm_ask_returns_false(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify fallback returns False when user declines."""

        def mock_ask(*args: object, **kwargs: object) -> bool:
            return False

        monkeypatch.setattr("rich.prompt.Confirm.ask", mock_ask)
        requester = RichPermissionRequester(Console(force_terminal=True))
        call = ToolCall(ToolName("bash"), ToolInput('echo "hello"'))
        assert requester.request_approval(call) is False
