"""Visitor that advances the loop per decision type, replacing `kind` branches."""

from __future__ import annotations

import time
from dataclasses import dataclass

from little_harness.application.agent_dependencies import AgentDependencies
from little_harness.application.decision_appliers import (
    MessageInjectingApplier,
    PreToolDecisionApplier,
    StopDecisionApplier,
)
from little_harness.application.loop_state import AgentLoopState
from little_harness.application.ports.agent_tool import AgentTool
from little_harness.domain.decision import FinalAnswer, ToolCall
from little_harness.domain.step import AgentStep
from little_harness.domain.tool_result import ToolRunRequest, ToolRunResult
from little_harness.domain.values.numeric_values import ElapsedSeconds, Iteration
from little_harness.domain.values.text_values import (
    MessageContent,
    Prompt,
    RunId,
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
