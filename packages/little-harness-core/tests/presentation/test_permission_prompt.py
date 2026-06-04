"""The interactive requester shows the call and reads a yes/no answer."""

from __future__ import annotations

from io import StringIO

import pytest
from little_harness.domain.decision import ToolCall
from little_harness.domain.values.text_values import ToolInput, ToolName
from little_harness.presentation.cli.permission_prompt import (
    InteractivePermissionRequester,
)

CALL = ToolCall(ToolName("bash"), ToolInput("ls -la"))


def requester_reading(
    answer: str,
) -> tuple[InteractivePermissionRequester, StringIO]:
    output = StringIO()
    requester = InteractivePermissionRequester(output=output, source=StringIO(answer))
    return requester, output


class TestInteractivePermissionRequester:
    @pytest.mark.parametrize("answer", ["y", "yes", "Y", "  Yes  "])
    def test_approves_on_an_affirmative_answer(self, answer: str) -> None:
        # Arrange
        requester, _ = requester_reading(answer)

        # Act / Assert
        assert requester.request_approval(CALL) is True

    @pytest.mark.parametrize("answer", ["n", "no", "maybe"])
    def test_rejects_on_any_other_answer(self, answer: str) -> None:
        # Arrange
        requester, _ = requester_reading(answer)

        # Act / Assert
        assert requester.request_approval(CALL) is False

    def test_rejects_when_input_is_closed(self) -> None:
        # Arrange: a closed stream reads as empty, which must deny, not crash.
        requester, _ = requester_reading("")

        # Act / Assert
        assert requester.request_approval(CALL) is False

    def test_shows_the_tool_name_and_input_in_the_prompt(self) -> None:
        # Arrange
        requester, output = requester_reading("y")

        # Act
        requester.request_approval(CALL)

        # Assert: the operator sees exactly what they are approving.
        prompt = output.getvalue()
        assert "'bash'" in prompt
        assert "'ls -la'" in prompt
