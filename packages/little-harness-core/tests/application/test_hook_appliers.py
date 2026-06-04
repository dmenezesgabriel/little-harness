"""The applier visitors turn a HookDecision into a concrete loop effect."""

from __future__ import annotations

from little_harness.application.agent_runtime import SessionDecisionApplier
from little_harness.application.decision_handler import (
    MessageInjectingApplier,
    PreToolDecisionApplier,
    StopDecisionApplier,
)
from little_harness.application.loop_state import AgentLoopState
from little_harness.domain.hook_decision import Block, InjectContext, Proceed
from little_harness.domain.message import ChatMessage
from little_harness.domain.message_history import MessageHistory
from little_harness.domain.tool_result import ToolRunResult
from little_harness.domain.values.role import SYSTEM, USER
from little_harness.domain.values.text_values import (
    MessageContent,
    ToolName,
    ToolOutput,
)


def empty_state() -> AgentLoopState:
    return AgentLoopState(MessageHistory())


class TestSessionDecisionApplier:
    def test_proceed_returns_none_and_adds_no_message(self) -> None:
        state = empty_state()

        result = Proceed().accept(SessionDecisionApplier(state, SYSTEM))

        assert result is None
        assert list(state.messages) == []

    def test_inject_appends_a_message_with_the_given_role(self) -> None:
        state = empty_state()

        result = InjectContext(MessageContent("ctx")).accept(
            SessionDecisionApplier(state, SYSTEM)
        )

        assert result is None
        assert list(state.messages) == [ChatMessage(SYSTEM, MessageContent("ctx"))]

    def test_block_returns_the_reason_and_adds_no_message(self) -> None:
        state = empty_state()

        result = Block(MessageContent("denied")).accept(
            SessionDecisionApplier(state, SYSTEM)
        )

        assert result == MessageContent("denied")
        assert list(state.messages) == []


class TestPreToolDecisionApplier:
    def test_proceed_returns_none_to_run_the_tool(self) -> None:
        state = empty_state()

        result = Proceed().accept(PreToolDecisionApplier(state, ToolName("calculator")))

        assert result is None
        assert list(state.messages) == []

    def test_inject_appends_a_user_message_and_runs_the_tool(self) -> None:
        state = empty_state()

        result = InjectContext(MessageContent("hint")).accept(
            PreToolDecisionApplier(state, ToolName("calculator"))
        )

        assert result is None
        assert list(state.messages) == [ChatMessage(USER, MessageContent("hint"))]

    def test_block_returns_a_failed_result_carrying_the_reason(self) -> None:
        state = empty_state()

        result = Block(MessageContent("not allowed")).accept(
            PreToolDecisionApplier(state, ToolName("calculator"))
        )

        assert result == ToolRunResult(
            ToolName("calculator"), ToolOutput("not allowed"), succeeded=False
        )
        assert list(state.messages) == []


class TestMessageInjectingApplier:
    def test_proceed_adds_no_message(self) -> None:
        state = empty_state()

        Proceed().accept(MessageInjectingApplier(state))

        assert list(state.messages) == []

    def test_inject_appends_a_user_message(self) -> None:
        state = empty_state()

        InjectContext(MessageContent("note")).accept(MessageInjectingApplier(state))

        assert list(state.messages) == [ChatMessage(USER, MessageContent("note"))]

    def test_block_appends_the_reason_as_a_user_message(self) -> None:
        state = empty_state()

        Block(MessageContent("rejected")).accept(MessageInjectingApplier(state))

        assert list(state.messages) == [ChatMessage(USER, MessageContent("rejected"))]


class TestStopDecisionApplier:
    def test_proceed_returns_the_answer_and_stops(self) -> None:
        state = empty_state()

        result = Proceed().accept(StopDecisionApplier(state, MessageContent("ans")))

        assert result == MessageContent("ans")
        assert list(state.messages) == []

    def test_inject_appends_a_message_and_still_returns_the_answer(self) -> None:
        state = empty_state()

        result = InjectContext(MessageContent("fyi")).accept(
            StopDecisionApplier(state, MessageContent("ans"))
        )

        assert result == MessageContent("ans")
        assert list(state.messages) == [ChatMessage(USER, MessageContent("fyi"))]

    def test_block_appends_the_reason_and_returns_none_to_keep_looping(self) -> None:
        state = empty_state()

        result = Block(MessageContent("keep going")).accept(
            StopDecisionApplier(state, MessageContent("ans"))
        )

        assert result is None
        assert list(state.messages) == [ChatMessage(USER, MessageContent("keep going"))]
