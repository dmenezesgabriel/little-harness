"""Visitor that advances the loop per decision type, replacing `kind` branches."""

from __future__ import annotations

import time
from dataclasses import dataclass

from local_llm.application.agent_dependencies import AgentDependencies
from local_llm.application.loop_state import AgentLoopState
from local_llm.application.ports.agent_tool import AgentTool
from local_llm.domain.decision import FinalAnswer, ToolCall
from local_llm.domain.step import AgentStep
from local_llm.domain.tool_result import ToolRunRequest, ToolRunResult
from local_llm.domain.values.numeric_values import ElapsedSeconds, Iteration
from local_llm.domain.values.text_values import (
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
        self._dependencies = dependencies
        self._context = context

    def visit_final_answer(self, decision: FinalAnswer) -> MessageContent | None:
        return decision.answer

    def visit_tool_call(self, decision: ToolCall) -> MessageContent | None:
        started_at = time.perf_counter()
        result = self._execute(decision)
        elapsed = ElapsedSeconds(time.perf_counter() - started_at)
        self._dependencies.observer.on_tool_invoked(
            self._context.run_id, self._context.iteration, result, elapsed
        )
        message = self._dependencies.policy.build_tool_observation_message(
            self._context.prompt, result
        )
        self._record(decision, result)
        self._context.state.append_message(message)
        return None

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
            return tool.run(request)
        except Exception as error:  # tool failures become observations, not crashes
            return ToolRunResult(
                decision.tool_name,
                ToolOutput(f"Tool error: {error}"),
                succeeded=False,
            )

    def _record(self, decision: ToolCall, result: ToolRunResult) -> None:
        step = AgentStep(
            self._context.iteration,
            self._context.model_output,
            decision,
            MessageContent(result.output.value),
        )
        self._context.state.record_step(step)
