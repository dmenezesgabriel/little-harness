"""Visitor that advances the loop per decision type, replacing `kind` branches."""

from __future__ import annotations

import time
from dataclasses import dataclass

from little_harness.application.agent_dependencies import AgentDependencies
from little_harness.application.loop_state import AgentLoopState
from little_harness.application.ports.agent_tool import AgentTool
from little_harness.domain.decision import FinalAnswer, ToolCall
from little_harness.domain.hook_decision import Block, InjectContext, Proceed
from little_harness.domain.message import ChatMessage
from little_harness.domain.step import AgentStep
from little_harness.domain.tool_result import ToolRunRequest, ToolRunResult
from little_harness.domain.values.numeric_values import ElapsedSeconds, Iteration
from little_harness.domain.values.role import USER
from little_harness.domain.values.text_values import (
    MessageContent,
    Prompt,
    RunId,
    ToolName,
    ToolOutput,
)


@dataclass(frozen=True)
class IterationContext:
    """Everything the visitor needs about the current iteration."""

    run_id: RunId
    prompt: Prompt
    iteration: Iteration
    model_output: MessageContent
    state: AgentLoopState


class LoopDecisionVisitor:
    """Returns the final answer for a `FinalAnswer`, or None to keep looping.

    For a `ToolCall` it runs the tool, records the step, appends the observation
    message, and returns None so the loop continues. Holds exactly two fields.
    """

    def __init__(
        self,
        dependencies: AgentDependencies,
        context: IterationContext,
    ) -> None:
        """See class docstring for argument descriptions."""
        self._dependencies = dependencies
        self._context = context

    def visit_final_answer(self, decision: FinalAnswer) -> MessageContent | None:
        """Apply stop hooks and return the answer or None to keep looping."""
        stop = self._dependencies.hooks.on_stop(
            self._context.run_id, self._context.iteration, decision.answer
        )
        return stop.accept(StopDecisionApplier(self._context.state, decision.answer))

    def visit_tool_call(self, decision: ToolCall) -> MessageContent | None:
        """Execute the tool, record the step, and return None to continue."""
        result = self._resolve_tool_result(decision)
        message = self._dependencies.policy.build_tool_observation_message(
            self._context.prompt, result
        )
        self._record(decision, result)
        self._context.state.append_message(message)
        return None

    def _resolve_tool_result(self, decision: ToolCall) -> ToolRunResult:
        blocked = self._pre_tool_use(decision)

        if blocked is not None:
            return blocked

        result = self._invoke(decision)
        self._post_tool_use(decision, result)
        return result

    def _pre_tool_use(self, decision: ToolCall) -> ToolRunResult | None:
        pre = self._dependencies.hooks.on_pre_tool_use(
            self._context.run_id, self._context.iteration, decision
        )
        applier = PreToolDecisionApplier(self._context.state, decision.tool_name)
        return pre.accept(applier)

    def _post_tool_use(self, decision: ToolCall, result: ToolRunResult) -> None:
        post = self._dependencies.hooks.on_post_tool_use(
            self._context.run_id, self._context.iteration, decision, result
        )
        post.accept(MessageInjectingApplier(self._context.state))

    def _invoke(self, decision: ToolCall) -> ToolRunResult:
        started_at = time.perf_counter()
        result = self._execute(decision)
        elapsed = ElapsedSeconds(time.perf_counter() - started_at)
        self._dependencies.observer.on_tool_invoked(
            self._context.run_id, self._context.iteration, result, elapsed
        )
        return result

    def _execute(self, decision: ToolCall) -> ToolRunResult:
        tool = self._dependencies.tool_registry.find(decision.tool_name)

        if tool is None:
            return ToolRunResult(
                decision.tool_name,
                ToolOutput(
                    f"Unknown tool: {decision.tool_name.value}. "
                    "Expected one registered tool."
                ),
                succeeded=False,
            )

        return self._run_safely(tool, decision)

    def _run_safely(self, tool: AgentTool, decision: ToolCall) -> ToolRunResult:
        request = ToolRunRequest(decision.tool_name, decision.tool_input)

        try:
            result = tool.run(request)
        except Exception as error:  # tool failures become observations, not crashes
            return ToolRunResult(
                decision.tool_name,
                ToolOutput(f"Tool error: {error}"),
                succeeded=False,
            )

        if result.succeeded:
            truncated = self._dependencies.truncator.truncate(
                result.output.value, self._dependencies.truncation_config
            )
            if truncated.truncated:
                return ToolRunResult(
                    decision.tool_name,
                    ToolOutput(truncated.content),
                    succeeded=True,
                )

        return result

    def _record(self, decision: ToolCall, result: ToolRunResult) -> None:
        step = AgentStep(
            self._context.iteration,
            self._context.model_output,
            decision,
            MessageContent(result.output.value),
        )
        self._context.state.record_step(step)


class PreToolDecisionApplier:
    """Applies a pre-tool hook decision; a block returns the result to use instead.

    Implements `HookDecisionVisitor[ToolRunResult | None]`: None means run the
    tool, a `ToolRunResult` means skip it and treat the reason as a failure.
    """

    def __init__(self, state: AgentLoopState, tool_name: ToolName) -> None:
        """See class docstring for argument descriptions."""
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
        """See class docstring for argument descriptions."""
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
    """Applies a model-request hook decision; Block skips the API call.

    Implements `HookDecisionVisitor[MessageContent | None]`: None means call
    the model, a `MessageContent` means use it as fake model output.
    """

    def __init__(self, state: AgentLoopState) -> None:
        """See class docstring for argument descriptions."""
        self._state = state

    def visit_proceed(self, _decision: Proceed) -> MessageContent | None:
        """Call the model normally."""
        return None

    def visit_inject_context(
        self, decision: InjectContext
    ) -> MessageContent | None:
        """Inject context and call the model."""
        self._state.append_message(ChatMessage(USER, decision.content))
        return None

    def visit_block(self, decision: Block) -> MessageContent | None:
        """Skip the model call and use the reason as output."""
        return decision.reason


class ContextBuildApplier:
    """Applies a context-build hook decision; Block skips the API call.

    Implements ``HookDecisionVisitor[MessageContent | None]``: None means call
    the model, a ``MessageContent`` means use it as fake model output.
    """

    def __init__(self, state: AgentLoopState) -> None:
        """See class docstring for argument descriptions."""
        self._state = state

    def visit_proceed(self, _decision: Proceed) -> MessageContent | None:
        """Call the model normally."""
        return None

    def visit_inject_context(
        self, decision: InjectContext
    ) -> MessageContent | None:
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
        """See class docstring for argument descriptions."""
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
        """See class docstring for argument descriptions."""
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
