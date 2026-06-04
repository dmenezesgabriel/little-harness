"""The chain folds its hooks: blocks short-circuit, injects concatenate."""

from __future__ import annotations

from local_llm.application.hook_chain import HookChain
from local_llm.domain.decision import ToolCall
from local_llm.domain.hook_decision import Block, InjectContext, Proceed
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
from tests.application.fakes import ScriptedHook

RUN_ID = RunId("run")
PROMPT = Prompt("question")
ITERATION = Iteration(1)
CALL = ToolCall(ToolName("calculator"), ToolInput("2 + 2"))
TOOL_RESULT = ToolRunResult(ToolName("calculator"), ToolOutput("4"), succeeded=True)
ANSWER = MessageContent("done")
RUN_RESULT = AgentResult(ANSWER, ElapsedSeconds(0.0), AgentSteps())


class TestHookChainFold:
    def test_proceeds_when_every_hook_proceeds(self) -> None:
        chain = HookChain([ScriptedHook(), ScriptedHook()])

        assert chain.on_session_start(RUN_ID, PROMPT) == Proceed()

    def test_passes_a_single_injection_through(self) -> None:
        injecting = ScriptedHook(session_start=InjectContext(MessageContent("a")))
        chain = HookChain([injecting])

        assert chain.on_session_start(RUN_ID, PROMPT) == InjectContext(
            MessageContent("a")
        )

    def test_concatenates_multiple_injections_in_order(self) -> None:
        first = ScriptedHook(session_start=InjectContext(MessageContent("a")))
        second = ScriptedHook(session_start=InjectContext(MessageContent("b")))
        chain = HookChain([first, second])

        assert chain.on_session_start(RUN_ID, PROMPT) == InjectContext(
            MessageContent("a\nb")
        )

    def test_block_short_circuits_before_later_hooks(self) -> None:
        blocking = ScriptedHook(session_start=Block(MessageContent("no")))
        later = ScriptedHook()
        chain = HookChain([blocking, later])

        decision = chain.on_session_start(RUN_ID, PROMPT)

        assert decision == Block(MessageContent("no"))
        assert later.calls == []

    def test_a_proceeding_hook_does_not_short_circuit_the_chain(self) -> None:
        # A proceed must let later hooks run, so a later injection still lands.
        first = ScriptedHook(session_start=Proceed())
        second = ScriptedHook(session_start=InjectContext(MessageContent("b")))
        chain = HookChain([first, second])

        assert chain.on_session_start(RUN_ID, PROMPT) == InjectContext(
            MessageContent("b")
        )

    def test_each_method_delegates_with_its_arguments_and_folds(self) -> None:
        # Every method must call the matching hook method with the right args and
        # return the folded decision, not just session_start.
        hook = ScriptedHook(
            session_start=InjectContext(MessageContent("s")),
            user_prompt_submit=InjectContext(MessageContent("u")),
            pre_tool_use=InjectContext(MessageContent("pre")),
            post_tool_use=InjectContext(MessageContent("post")),
            stop=InjectContext(MessageContent("stop")),
        )
        chain = HookChain([hook])

        assert chain.on_session_start(RUN_ID, PROMPT) == InjectContext(
            MessageContent("s")
        )
        assert chain.on_user_prompt_submit(RUN_ID, PROMPT) == InjectContext(
            MessageContent("u")
        )
        assert chain.on_pre_tool_use(RUN_ID, ITERATION, CALL) == InjectContext(
            MessageContent("pre")
        )
        assert chain.on_post_tool_use(
            RUN_ID, ITERATION, CALL, TOOL_RESULT
        ) == InjectContext(MessageContent("post"))
        assert chain.on_stop(RUN_ID, ITERATION, ANSWER) == InjectContext(
            MessageContent("stop")
        )
        chain.on_session_end(RUN_ID, RUN_RESULT)

        assert hook.run_ids == [RUN_ID] * 6
        assert hook.prompts == [PROMPT, PROMPT]
        assert hook.iterations == [ITERATION, ITERATION, ITERATION]
        assert hook.tool_calls == [CALL, CALL]
        assert hook.tool_results == [TOOL_RESULT]
        assert hook.answers == [ANSWER]
        assert hook.ended_with == [RUN_RESULT]

    def test_session_end_fans_out_to_every_hook(self) -> None:
        first = ScriptedHook()
        second = ScriptedHook()
        chain = HookChain([first, second])
        result = AgentResult(MessageContent("done"), ElapsedSeconds(0.0), AgentSteps())

        chain.on_session_end(RUN_ID, result)

        assert first.calls == ["session_end"]
        assert second.calls == ["session_end"]
