# pyright: reportPrivateUsage=false
from __future__ import annotations

import pytest
from little_harness.domain.decision import ToolCall
from little_harness.domain.values.text_values import ToolInput, ToolName
from little_harness_rich.permission import RichPermissionRequester
from rich.console import Console


class TestRichPermissionRequester:
    def test_request_approval_returns_true_when_confirmed_and_passes_correct_args(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        call_args = {}

        def mock_ask(*args: object, **kwargs: object) -> bool:
            call_args["args"] = args
            call_args["kwargs"] = kwargs
            return True

        monkeypatch.setattr("rich.prompt.Confirm.ask", mock_ask)
        console = Console(force_terminal=True)
        requester = RichPermissionRequester(console)

        assert requester._console is console

        call = ToolCall(ToolName("bash"), ToolInput('echo "hello"'))

        assert requester.request_approval(call) is True

        # Assert args to kill mutmut
        expected_prompt = (
            "Allow tool [cyan]'bash'[/cyan] to run with input "
            "[yellow]'echo \"hello\"'[/yellow]?"
        )
        assert call_args["args"] == (expected_prompt,)
        assert call_args["kwargs"] == {"console": console, "default": False}

    def test_request_approval_returns_false_when_denied(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def mock_ask(*args: object, **kwargs: object) -> bool:
            return False

        monkeypatch.setattr("rich.prompt.Confirm.ask", mock_ask)
        requester = RichPermissionRequester(Console(force_terminal=True))
        call = ToolCall(ToolName("bash"), ToolInput('echo "hello"'))
        assert requester.request_approval(call) is False
