"""The auto-approve requester grants every call for unattended runs."""

from __future__ import annotations

from little_harness.domain.decision import ToolCall
from little_harness.domain.values.text_values import ToolInput, ToolName
from little_harness.infrastructure.permissions.auto_approve_requester import (
    AutoApprovePermissionRequester,
)


class TestAutoApprovePermissionRequester:
    def test_approves_any_call(self) -> None:
        # Arrange
        call = ToolCall(ToolName("bash"), ToolInput("rm file"))

        # Act / Assert
        assert AutoApprovePermissionRequester().request_approval(call) is True
