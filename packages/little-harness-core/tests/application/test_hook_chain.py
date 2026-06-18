"""The chain folds its hooks: blocks short-circuit, injects concatenate."""

from __future__ import annotations

from little_harness.application.hook_chain import HookChain
from little_harness.domain.decision import ToolCall
from little_harness.domain.hook_decision import Block, InjectContext, Proceed
from little_harness.domain.message_history import MessageHistory
from little_harness.domain.result import AgentResult
from little_harness.domain.steps import AgentSteps
from little_harness.domain.tool_result import ToolRunResult
from little_harness.domain.values.numeric_values import ElapsedSeconds, Iteration
from little_harness.domain.values.text_values import (
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


def _invoke_all_hook_methods(chain: HookChain) -> None:
    """Call every HookChain method once with fixed test arguments."""
    chain.on_session_start(RUN_ID, PROMPT)
    chain.on_user_prompt_submit(RUN_ID, PROMPT)
    chain.on_turn_start(RUN_ID, ITERATION, PROMPT)
    chain.on_turn_end(RUN_ID, ITERATION, ANSWER)
    chain.on_model_request(RUN_ID, ITERATION)
    chain.on_model_response(RUN_ID, ITERATION, ANSWER)
    chain.on_context_build(RUN_ID, ITERATION, MessageHistory())
    chain.on_pre_tool_use(RUN_ID, ITERATION, CALL)
    chain.on_post_tool_use(RUN_ID, ITERATION, CALL, TOOL_RESULT)
    chain.on_stop(RUN_ID, ITERATION, ANSWER)
    chain.on_session_end(RUN_ID, RUN_RESULT)


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

    def test_each_method_delegates_and_folds(self) -> None:
        # Every method must call the matching hook method with the right args and
        # return the folded decision, not just session_start.
        hook = ScriptedHook(
            session_start=InjectContext(MessageContent("s")),
            user_prompt_submit=InjectContext(MessageContent("u")),
            turn_start=InjectContext(MessageContent("ts")),
            turn_end=InjectContext(MessageContent("te")),
            model_request=InjectContext(MessageContent("mr")),
            model_response=InjectContext(MessageContent("mrs")),
            context_build=InjectContext(MessageContent("cb")),
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
        assert chain.on_turn_start(RUN_ID, ITERATION, PROMPT) == InjectContext(
            MessageContent("ts")
        )
        assert chain.on_turn_end(RUN_ID, ITERATION, ANSWER) == InjectContext(
            MessageContent("te")
        )
        assert chain.on_model_request(RUN_ID, ITERATION) == InjectContext(
            MessageContent("mr")
        )
        assert chain.on_model_response(RUN_ID, ITERATION, ANSWER) == InjectContext(
            MessageContent("mrs")
        )
        assert chain.on_context_build(
            RUN_ID, ITERATION, MessageHistory()
        ) == InjectContext(MessageContent("cb"))
        assert chain.on_pre_tool_use(RUN_ID, ITERATION, CALL) == InjectContext(
            MessageContent("pre")
        )
        assert chain.on_post_tool_use(
            RUN_ID, ITERATION, CALL, TOOL_RESULT
        ) == InjectContext(MessageContent("post"))
        assert chain.on_stop(RUN_ID, ITERATION, ANSWER) == InjectContext(
            MessageContent("stop")
        )

    def test_each_method_records_the_correct_arguments(self) -> None:
        hook = ScriptedHook(
            session_start=InjectContext(MessageContent("s")),
            user_prompt_submit=InjectContext(MessageContent("u")),
            turn_start=InjectContext(MessageContent("ts")),
            turn_end=InjectContext(MessageContent("te")),
            model_request=InjectContext(MessageContent("mr")),
            model_response=InjectContext(MessageContent("mrs")),
            context_build=InjectContext(MessageContent("cb")),
            pre_tool_use=InjectContext(MessageContent("pre")),
            post_tool_use=InjectContext(MessageContent("post")),
            stop=InjectContext(MessageContent("stop")),
        )
        _invoke_all_hook_methods(HookChain([hook]))

        assert hook.run_ids == [RUN_ID] * 11
        assert hook.prompts == [PROMPT, PROMPT, PROMPT]
        assert hook.iterations == [ITERATION] * 8
        assert hook.tool_calls == [CALL, CALL]
        assert hook.tool_results == [TOOL_RESULT]
        assert hook.answers == [ANSWER]
        assert hook.outputs == [ANSWER, ANSWER]
        assert hook.ended_with == [RUN_RESULT]

    def test_session_end_fans_out_to_every_hook(self) -> None:
        first = ScriptedHook()
        second = ScriptedHook()
        chain = HookChain([first, second])
        result = AgentResult(MessageContent("done"), ElapsedSeconds(0.0), AgentSteps())

        chain.on_session_end(RUN_ID, result)

        assert first.calls == ["session_end"]
        assert second.calls == ["session_end"]
