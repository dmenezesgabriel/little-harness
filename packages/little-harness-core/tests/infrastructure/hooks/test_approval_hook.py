"""The approval hook gates only sensitive tools and blocks on rejection."""

from __future__ import annotations

from little_harness.application.ports.lifecycle_hook import LifecycleHook
from little_harness.domain.decision import ToolCall
from little_harness.domain.hook_decision import Block, Proceed
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
from little_harness.infrastructure.hooks.approval_hook import ApprovalHook

RUN_ID = RunId("run")
ITERATION = Iteration(1)
PROMPT = Prompt("question")
ANSWER = MessageContent("done")
BASH_CALL = ToolCall(ToolName("bash"), ToolInput("ls"))
READ_CALL = ToolCall(ToolName("read_file"), ToolInput("README.md"))
RESULT = ToolRunResult(ToolName("bash"), ToolOutput("ok"), succeeded=True)
RUN_RESULT = AgentResult(ANSWER, ElapsedSeconds(0.0), AgentSteps())


class RecordingRequester:
    """Named PermissionRequester double with a preset verdict and a call log."""

    def __init__(self, approve: bool) -> None:
        self._approve = approve
        self.calls: list[ToolCall] = []

    def request_approval(self, call: ToolCall) -> bool:
        self.calls.append(call)
        return self._approve


class TestApprovalHook:
    def test_proceeds_without_asking_for_a_tool_that_needs_no_approval(self) -> None:
        # Arrange
        requester = RecordingRequester(approve=False)
        hook = ApprovalHook(requester, frozenset({"bash"}))

        # Act
        decision = hook.on_pre_tool_use(RUN_ID, ITERATION, READ_CALL)

        # Assert: a safe tool is never put in front of the operator.
        assert decision == Proceed()
        assert requester.calls == []

    def test_proceeds_when_the_operator_approves(self) -> None:
        # Arrange
        requester = RecordingRequester(approve=True)
        hook = ApprovalHook(requester, frozenset({"bash"}))

        # Act
        decision = hook.on_pre_tool_use(RUN_ID, ITERATION, BASH_CALL)

        # Assert
        assert decision == Proceed()
        assert requester.calls == [BASH_CALL]

    def test_blocks_with_a_reason_when_the_operator_rejects(self) -> None:
        # Arrange
        requester = RecordingRequester(approve=False)
        hook = ApprovalHook(requester, frozenset({"bash"}))

        # Act
        decision = hook.on_pre_tool_use(RUN_ID, ITERATION, BASH_CALL)

        # Assert: the model learns why, so it can choose another path.
        assert isinstance(decision, Block)
        assert "'bash'" in decision.reason.value
        assert "not approved" in decision.reason.value


class TestApprovalHookLeavesOtherPointsUntouched:
    def test_conforms_to_the_lifecycle_hook_port(self) -> None:
        # Arrange: the annotation forces a protocol-conformance check.
        hook: LifecycleHook = ApprovalHook(
            RecordingRequester(approve=True), frozenset()
        )

        assert hook is not None

    def test_proceeds_at_every_point_other_than_pre_tool_use(self) -> None:
        # Arrange
        hook = ApprovalHook(RecordingRequester(approve=True), frozenset())

        # Act / Assert: only pre-tool-use gates; the rest never block a run.
        assert hook.on_session_start(RUN_ID, PROMPT) == Proceed()
        assert hook.on_user_prompt_submit(RUN_ID, PROMPT) == Proceed()
        assert hook.on_post_tool_use(RUN_ID, ITERATION, BASH_CALL, RESULT) == Proceed()
        assert hook.on_stop(RUN_ID, ITERATION, ANSWER) == Proceed()
        assert hook.on_session_end(RUN_ID, RUN_RESULT) is None
