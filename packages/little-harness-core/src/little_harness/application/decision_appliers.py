"""Visitor implementations that turn a HookDecision into a concrete loop effect.

Every lifecycle hook point yields a `HookDecision`; these appliers translate
that decision into a mutation of the agent loop state -- injecting a message,
blocking with a reason, or returning the original output unchanged.

Example:
    blocked = decision.accept(SessionDecisionApplier(state, SYSTEM))

"""

from __future__ import annotations

from little_harness.application.loop_state import AgentLoopState
from little_harness.domain.hook_decision import Block, InjectContext, Proceed
from little_harness.domain.message import ChatMessage
from little_harness.domain.tool_result import ToolRunResult
from little_harness.domain.values.role import USER, Role
from little_harness.domain.values.text_values import (
    MessageContent,
    ToolName,
    ToolOutput,
)


class SessionDecisionApplier:
    """Applies a session hook's decision: inject a message, or report a block.

    Implements `HookDecisionVisitor[MessageContent | None]`; the returned reason
    is the answer the run aborts with, or None to continue.
    """

    def __init__(self, state: AgentLoopState, role: Role) -> None:
        """Store the loop state and role for message injection."""
        self._state = state
        self._role = role

    def visit_proceed(self, _decision: Proceed) -> MessageContent | None:
        """Continue without injecting or blocking."""
        return None

    def visit_inject_context(self, decision: InjectContext) -> MessageContent | None:
        """Inject the context message under the configured role."""
        self._state.append_message(ChatMessage(self._role, decision.content))
        return None

    def visit_block(self, decision: Block) -> MessageContent | None:
        """Abort the run with the block reason."""
        return decision.reason


class PreToolDecisionApplier:
    """Applies a pre-tool hook decision; a block returns the result to use instead.

    Implements `HookDecisionVisitor[ToolRunResult | None]`: None means run the
    tool, a `ToolRunResult` means skip it and treat the reason as a failure.
    """

    def __init__(self, state: AgentLoopState, tool_name: ToolName) -> None:
        """Store the loop state and tool name for pre-tool decisions."""
        self._state = state
        self._tool_name = tool_name

    def visit_proceed(self, _decision: Proceed) -> ToolRunResult | None:
        """Continue without injecting or blocking."""
        return None

    def visit_inject_context(self, decision: InjectContext) -> ToolRunResult | None:
        """Inject context and continue."""
        self._state.append_message(ChatMessage(USER, decision.content))
        return None

    def visit_block(self, decision: Block) -> ToolRunResult | None:
        """Return a failed tool result with the block reason."""
        output = ToolOutput(decision.reason.value)
        return ToolRunResult(self._tool_name, output, succeeded=False)


class MessageInjectingApplier:
    """Appends a user message for inject/block; does nothing on proceed.

    Implements `HookDecisionVisitor[None]` for points where the action already
    happened (post-tool), so both inject and block only add feedback.
    """

    def __init__(self, state: AgentLoopState) -> None:
        """Store the loop state for post-tool message injection."""
        self._state = state

    def visit_proceed(self, _decision: Proceed) -> None:
        """No-op: nothing to inject."""

    def visit_inject_context(self, decision: InjectContext) -> None:
        """Append the context as a user message."""
        self._state.append_message(ChatMessage(USER, decision.content))

    def visit_block(self, decision: Block) -> None:
        """Append the block reason as a user message."""
        self._state.append_message(ChatMessage(USER, decision.reason))


class ModelRequestApplier:
    """Applies a model-request or context-build hook decision.

    Block skips the API call and returns the reason as fake model output.
    None means call the model normally. Used for both ``on_model_request``
    and ``on_context_build`` which share identical semantics.

    Implements `HookDecisionVisitor[MessageContent | None]`.
    """

    def __init__(self, state: AgentLoopState) -> None:
        """Store the loop state for blocking or injecting."""
        self._state = state

    def visit_proceed(self, _decision: Proceed) -> MessageContent | None:
        """Call the model normally."""
        return None

    def visit_inject_context(self, decision: InjectContext) -> MessageContent | None:
        """Inject context and call the model."""
        self._state.append_message(ChatMessage(USER, decision.content))
        return None

    def visit_block(self, decision: Block) -> MessageContent | None:
        """Skip the model call and use the reason as output."""
        return decision.reason


class OutputReplacingApplier:
    """Applies a hook decision where Block replaces the output.

    Used by on_turn_end and on_model_response. Returns the original output
    unchanged for Proceed/InjectContext, or the block reason as replacement.

    Implements `HookDecisionVisitor[MessageContent]`.
    """

    def __init__(self, state: AgentLoopState, original: MessageContent) -> None:
        """Store the loop state and original output for replacement."""
        self._state = state
        self._original = original

    def visit_proceed(self, _decision: Proceed) -> MessageContent:
        """Return the original output unchanged."""
        return self._original

    def visit_inject_context(self, decision: InjectContext) -> MessageContent:
        """Inject the context and return the original output."""
        self._state.append_message(ChatMessage(USER, decision.content))
        return self._original

    def visit_block(self, decision: Block) -> MessageContent:
        """Replace the output with the block reason."""
        return decision.reason


class StopDecisionApplier:
    """Applies a stop hook decision; a block keeps looping by returning None.

    Implements `HookDecisionVisitor[MessageContent | None]`: the answer means
    stop, None means the loop continues with the reason as guidance.
    """

    def __init__(self, state: AgentLoopState, answer: MessageContent) -> None:
        """Store the loop state and answer for stop decisions."""
        self._state = state
        self._answer = answer

    def visit_proceed(self, _decision: Proceed) -> MessageContent | None:
        """Return the answer to stop the loop."""
        return self._answer

    def visit_inject_context(self, decision: InjectContext) -> MessageContent | None:
        """Inject the context message and return the answer."""
        self._state.append_message(ChatMessage(USER, decision.content))
        return self._answer

    def visit_block(self, decision: Block) -> MessageContent | None:
        """Inject the block reason and return None to keep looping."""
        self._state.append_message(ChatMessage(USER, decision.reason))
        return None
