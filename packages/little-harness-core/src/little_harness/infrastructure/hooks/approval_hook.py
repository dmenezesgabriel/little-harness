"""Pre-tool hook that enforces human-in-the-loop approval for sensitive tools.

Tools that declare `requires_approval` are gated here: the hook asks the injected
`PermissionRequester` and turns a rejection into a `Block`, so the model receives
the refusal as an observation and can choose a different path. Every other
lifecycle point proceeds untouched, so safe tools are never put to the operator.

Implements `LifecycleHook` structurally (like the test `ScriptedHook`) rather
than subclassing the variadic `NullHook`, which strict typing cannot narrow.
"""

from __future__ import annotations

from little_harness.application.ports.permission_requester import PermissionRequester
from little_harness.domain.decision import ToolCall
from little_harness.domain.hook_decision import Block, HookDecision, Proceed
from little_harness.domain.message_history import MessageHistory
from little_harness.domain.result import AgentResult
from little_harness.domain.tool_result import ToolRunResult
from little_harness.domain.values.numeric_values import Iteration
from little_harness.domain.values.text_values import MessageContent, Prompt, RunId


class ApprovalHook:
    """Asks for approval before running any tool whose name needs it.

    Example:
        hook = ApprovalHook(requester, frozenset({"bash"}))

    """

    def __init__(
        self,
        requester: PermissionRequester,
        names_requiring_approval: frozenset[str],
    ) -> None:
        """See class docstring for argument descriptions."""
        self._requester = requester
        self._names_requiring_approval = names_requiring_approval

    def on_session_start(self, _run_id: RunId, _prompt: Prompt) -> HookDecision:
        """Proceed with the session start without approval check."""
        return Proceed()

    def on_user_prompt_submit(self, _run_id: RunId, _prompt: Prompt) -> HookDecision:
        """Proceed with the user prompt without approval check."""
        return Proceed()

    def on_turn_start(
        self, _run_id: RunId, _iteration: Iteration, _prompt: Prompt
    ) -> HookDecision:
        """Proceed with turn start without approval check."""
        return Proceed()

    def on_turn_end(
        self, _run_id: RunId, _iteration: Iteration, _output: MessageContent
    ) -> HookDecision:
        """Proceed with turn end without approval check."""
        return Proceed()

    def on_model_request(self, _run_id: RunId, _iteration: Iteration) -> HookDecision:
        """Proceed with model request without approval check."""
        return Proceed()

    def on_model_response(
        self, _run_id: RunId, _iteration: Iteration, _output: MessageContent
    ) -> HookDecision:
        """Proceed with model response without approval check."""
        return Proceed()

    def on_context_build(
        self, _run_id: RunId, _iteration: Iteration, _messages: MessageHistory
    ) -> HookDecision:
        """Proceed with context build without approval check."""
        return Proceed()

    def on_pre_tool_use(
        self, _run_id: RunId, _iteration: Iteration, call: ToolCall
    ) -> HookDecision:
        """Block unapproved tool calls; proceed with safe tools."""
        if call.tool_name.value not in self._names_requiring_approval:
            return Proceed()

        if self._requester.request_approval(call):
            return Proceed()

        return Block(
            MessageContent(
                f"Tool {call.tool_name.value!r} was not approved by the operator. "
                "Expected operator approval before running a sensitive tool."
            )
        )

    def on_post_tool_use(
        self,
        _run_id: RunId,
        _iteration: Iteration,
        _call: ToolCall,
        _result: ToolRunResult,
    ) -> HookDecision:
        """Proceed after tool use without additional checks."""
        return Proceed()

    def on_stop(
        self, _run_id: RunId, _iteration: Iteration, _answer: MessageContent
    ) -> HookDecision:
        """Proceed with the stop without approval check."""
        return Proceed()

    def on_session_end(self, _run_id: RunId, _result: AgentResult) -> None:
        """Proceed with session end without cleanup."""
        ...
