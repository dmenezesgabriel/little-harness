"""Permission requester that approves every call without asking a human.

The default when there is no terminal to prompt (piped input, CI) or when the
operator passes `--yes`. The Null Object of the permission seam: it keeps the
agent running unattended while the per-tool guardrails still apply.
"""

from __future__ import annotations

from little_harness.domain.decision import ToolCall


class AutoApprovePermissionRequester:
    """Grants every request. Used for non-interactive, unattended runs.

    Example:
        AutoApprovePermissionRequester().request_approval(call)  # True
    """

    def request_approval(self, _call: ToolCall) -> bool:
        return True
