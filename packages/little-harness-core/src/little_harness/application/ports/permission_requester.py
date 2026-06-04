"""Port for obtaining permission to run a sensitive tool call.

This is the human-in-the-loop seam: when a tool declares `requires_approval`,
the runtime asks a `PermissionRequester` before running it. The interactive
implementation prompts a person; non-interactive implementations decide without
one, so automated runs and tests never block on input.
"""

from __future__ import annotations

from typing import Protocol

from little_harness.domain.decision import ToolCall


class PermissionRequester(Protocol):
    # Positional-only (like LifecycleHook) so implementations may name or ignore
    # the argument without breaking structural conformance.
    def request_approval(self, call: ToolCall, /) -> bool:
        """Return True to allow the call, False to reject it.

        Example:
            allowed = requester.request_approval(call)
        """
        ...
