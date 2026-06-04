"""The default hook proceeds everywhere and ends the session silently."""

from __future__ import annotations

from local_llm.application.ports.lifecycle_hook import LifecycleHook
from local_llm.domain.decision import ToolCall
from local_llm.domain.hook_decision import Proceed
from local_llm.domain.result import AgentResult
from local_llm.domain.steps import AgentSteps
from local_llm.domain.tool_result import ToolRunResult
from local_llm.domain.values.numeric_values import ElapsedSeconds, Iteration
from local_llm.domain.values.text_values import (
    MessageContent,
    Prompt,
    RunId,
    ToolInput,
    ToolName,
    ToolOutput,
)
from local_llm.infrastructure.hooks.null_hook import NullHook

RUN_ID = RunId("run")
ITERATION = Iteration(1)
PROMPT = Prompt("question")
CALL = ToolCall(ToolName("calculator"), ToolInput("2 + 2"))
RESULT = ToolRunResult(ToolName("calculator"), ToolOutput("4"), succeeded=True)
ANSWER = MessageContent("done")


class TestNullHook:
    def test_satisfies_the_lifecycle_hook_port(self) -> None:
        hook: LifecycleHook = NullHook()

        assert hook is not None

    def test_proceeds_at_every_decision_point(self) -> None:
        hook = NullHook()

        assert hook.on_session_start(RUN_ID, PROMPT) == Proceed()
        assert hook.on_user_prompt_submit(RUN_ID, PROMPT) == Proceed()
        assert hook.on_pre_tool_use(RUN_ID, ITERATION, CALL) == Proceed()
        assert hook.on_post_tool_use(RUN_ID, ITERATION, CALL, RESULT) == Proceed()
        assert hook.on_stop(RUN_ID, ITERATION, ANSWER) == Proceed()

    def test_session_end_returns_none(self) -> None:
        agent_result = AgentResult(ANSWER, ElapsedSeconds(0.0), AgentSteps())

        assert NullHook().on_session_end(RUN_ID, agent_result) is None
