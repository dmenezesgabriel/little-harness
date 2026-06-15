from __future__ import annotations

from collections.abc import Sequence

from little_harness.application.agent_dependencies import AgentDependencies
from little_harness.application.agent_runtime import (
    FALLBACK_ANSWER,
    AgentRuntime,
    AgentRuntimeConfig,
)
from little_harness.application.ports.agent_tool import AgentTool
from little_harness.application.ports.chat_model import ChatModel, ResponseSchema
from little_harness.application.ports.lifecycle_hook import LifecycleHook
from little_harness.application.tool_registry import ToolRegistry
from little_harness.domain.hook_decision import Block, InjectContext
from little_harness.domain.message import ChatMessage
from little_harness.domain.message_history import MessageHistory
from little_harness.domain.tool_result import ToolRunRequest, ToolRunResult
from little_harness.domain.values.numeric_values import (
    Iteration,
    MaxIterations,
    MaxTokens,
    RepeatPenalty,
    Temperature,
    TopP,
)
from little_harness.domain.values.role import ASSISTANT, SYSTEM, USER
from little_harness.domain.values.text_values import (
    MessageContent,
    Prompt,
    RunId,
    ToolInput,
    ToolName,
    ToolOutput,
)
from little_harness.domain.values.truncation import TruncationConfig
from little_harness.infrastructure.hooks.null_hook import NullHook

from tests.application.fakes import (
    ChunkedChatModel,
    DecisionQueuePolicy,
    FailingAgentTool,
    IdentityTruncator,
    RecordingAgentTool,
    RecordingChatModel,
    RecordingObserver,
    RecordingSkillLoader,
    RecordingTokenSink,
    ScriptedHook,
    final_decision,
    tool_decision,
)


def create_runtime(
    chat_model: ChatModel,
    tools: Sequence[AgentTool],
    policy: DecisionQueuePolicy,
    observer: RecordingObserver | None = None,
    max_iterations: int = 3,
    token_sink: RecordingTokenSink | None = None,
    hooks: LifecycleHook | None = None,
    top_p: TopP | None = None,
    repeat_penalty: RepeatPenalty | None = None,
    skill_loader: RecordingSkillLoader | None = None,
) -> AgentRuntime:
    dependencies = AgentDependencies(
        chat_model=chat_model,
        tool_registry=ToolRegistry(tools),
        policy=policy,
        observer=observer or RecordingObserver(),
        token_sink=token_sink or RecordingTokenSink(),
        hooks=hooks or NullHook(),
        truncator=IdentityTruncator(),
        truncation_config=TruncationConfig(max_lines=2000, max_bytes=51200),
        skill_loader=skill_loader or RecordingSkillLoader(),
    )
    config = AgentRuntimeConfig(
        max_iterations=MaxIterations(max_iterations),
        temperature=Temperature(0.0),
        max_tokens=MaxTokens(128),
        top_p=top_p,
        repeat_penalty=repeat_penalty,
    )
    return AgentRuntime(dependencies, config)


class TestAgentRuntimeBuildSystemMessage:
    def test_returns_a_system_message_with_the_policy_prompt(self) -> None:
        chat_model = RecordingChatModel(["final"])
        policy = DecisionQueuePolicy([final_decision("done")])
        runtime = create_runtime(chat_model, [RecordingAgentTool()], policy)

        message = runtime.build_system_message()

        assert message.role == SYSTEM
        assert message.content == MessageContent("Tools: 1")


class TestAgentRuntimeFinalAnswer:
    def test_returns_final_answer_without_using_tools(self) -> None:
        # Arrange
        chat_model = RecordingChatModel(["final"])
        policy = DecisionQueuePolicy([final_decision("done")])
        tool = RecordingAgentTool()
        runtime = create_runtime(chat_model, [tool], policy)

        # Act
        result = runtime.run(Prompt("question"))

        # Assert
        assert result.answer == MessageContent("done")
        assert list(result.steps) == []
        assert tool.requests == []
        assert len(chat_model.requests) == 1
        expected_messages = (
            MessageHistory()
            .with_message(ChatMessage(SYSTEM, MessageContent("Tools: 1")))
            .with_message(ChatMessage(USER, MessageContent("question")))
        )
        assert chat_model.requests[0].messages == expected_messages
        assert chat_model.requests[0].temperature == Temperature(0.0)
        assert chat_model.requests[0].max_tokens == MaxTokens(128)
        assert chat_model.requests[0].top_p is None
        assert chat_model.requests[0].repeat_penalty is None

    def test_forwards_top_p_and_repeat_penalty_to_the_chat_request(self) -> None:
        # Arrange
        chat_model = RecordingChatModel(["final"])
        policy = DecisionQueuePolicy([final_decision("done")])
        runtime = create_runtime(
            chat_model,
            [],
            policy,
            top_p=TopP(0.5),
            repeat_penalty=RepeatPenalty(1.1),
        )

        # Act
        runtime.run(Prompt("question"))

        # Assert
        assert chat_model.requests[0].top_p == TopP(0.5)
        assert chat_model.requests[0].repeat_penalty == RepeatPenalty(1.1)

    def test_forwards_the_policy_schema_built_from_the_registered_tools(self) -> None:
        # Arrange: the policy derives its schema from the registry's specs.
        chat_model = RecordingChatModel(["final"])
        policy = DecisionQueuePolicy([final_decision("done")])
        runtime = create_runtime(chat_model, [RecordingAgentTool()], policy)

        # Act
        runtime.run(Prompt("question"))

        # Assert: the runtime passes the registry's one spec to the policy, and
        # the returned schema rides along on the completion request.
        assert policy.schema_tool_counts == [1]
        assert chat_model.requests[0].response_schema == ResponseSchema({"tools": 1})


class TestAgentRuntimeToolUse:
    def test_executes_registered_tool_and_returns_final_answer(self) -> None:
        # Arrange
        chat_model = RecordingChatModel(["tool", "final"])
        policy = DecisionQueuePolicy(
            [tool_decision("calculator", "2 + 2"), final_decision("4")]
        )
        tool = RecordingAgentTool()
        runtime = create_runtime(chat_model, [tool], policy)

        # Act
        result = runtime.run(Prompt("question"))

        # Assert
        assert result.answer == MessageContent("4")
        assert len(list(result.steps)) == 1
        assert tool.requests == [
            ToolRunRequest(ToolName("calculator"), ToolInput("2 + 2"))
        ]
        assert policy.tool_results == [
            ToolRunResult(ToolName("calculator"), ToolOutput("4"), succeeded=True)
        ]
        step = next(iter(result.steps))
        assert step.iteration == Iteration(1)
        assert step.model_output == MessageContent("tool")
        assert step.decision == tool_decision("calculator", "2 + 2")
        assert step.observation == MessageContent("4")
        second_request = list(chat_model.requests[1].messages)
        assert ChatMessage(ASSISTANT, MessageContent("tool")) in second_request
        assert ChatMessage(USER, MessageContent("question: 4")) in second_request

    def test_reports_unknown_tool_as_failed_observation(self) -> None:
        # Arrange
        chat_model = RecordingChatModel(["tool", "final"])
        policy = DecisionQueuePolicy(
            [tool_decision("missing", "2 + 2"), final_decision("fallback")]
        )
        runtime = create_runtime(chat_model, [], policy)

        # Act
        result = runtime.run(Prompt("question"))

        # Assert
        assert result.answer == MessageContent("fallback")
        assert policy.tool_results == [
            ToolRunResult(
                ToolName("missing"),
                ToolOutput("Unknown tool: missing. Expected one registered tool."),
                succeeded=False,
            )
        ]

    def test_converts_tool_exception_to_failed_observation(self) -> None:
        # Arrange
        chat_model = RecordingChatModel(["tool", "final"])
        policy = DecisionQueuePolicy(
            [tool_decision("calculator", "2 + 2"), final_decision("fallback")]
        )
        runtime = create_runtime(chat_model, [FailingAgentTool()], policy)

        # Act
        result = runtime.run(Prompt("question"))

        # Assert
        assert result.answer == MessageContent("fallback")
        assert policy.tool_results == [
            ToolRunResult(
                ToolName("calculator"),
                ToolOutput("Tool error: Forced failure for 2 + 2"),
                succeeded=False,
            )
        ]


class TestAgentRuntimeRepair:
    def test_repairs_invalid_model_output(self) -> None:
        # Arrange
        chat_model = RecordingChatModel(["invalid", "final"])
        policy = DecisionQueuePolicy([final_decision("done")])
        observer = RecordingObserver()
        runtime = create_runtime(chat_model, [], policy, observer)

        # Act
        result = runtime.run(Prompt("question"))

        # Assert
        assert result.answer == MessageContent("done")
        assert len(list(result.steps)) == 1
        assert len(policy.repair_errors) == 1
        step = next(iter(result.steps))
        assert step.decision is None
        assert step.iteration == Iteration(1)
        assert step.model_output == MessageContent("invalid")
        assert "question" in step.observation.value
        assert "Invalid model output" in step.observation.value
        assert any(
            message.content.value.startswith("Repair")
            for message in chat_model.requests[1].messages
        )
        # The repair event carries the iteration, the real error, and one run id.
        repair_iteration, repair_error = observer.repairs[0]
        assert repair_iteration == Iteration(1)
        assert repair_error is policy.repair_errors[0]
        assert all(isinstance(rid, RunId) and rid.value for rid in observer.run_ids)
        assert len(set(observer.run_ids)) == 1


class TestAgentRuntimeExhaustion:
    def test_returns_fallback_after_max_iterations(self) -> None:
        # Arrange
        chat_model = RecordingChatModel(["tool"])
        policy = DecisionQueuePolicy([tool_decision("calculator", "2 + 2")])
        runtime = create_runtime(
            chat_model,
            [RecordingAgentTool()],
            policy,
            max_iterations=1,
        )

        # Act
        result = runtime.run(Prompt("question"))

        # Assert
        upper_bound_seconds = 10.0
        assert result.answer == FALLBACK_ANSWER
        assert len(list(result.steps)) == 1
        assert 0.0 <= result.elapsed.value < upper_bound_seconds


class TestAgentRuntimeStreaming:
    def test_joins_streamed_chunks_and_emits_each_to_the_token_sink(self) -> None:
        # Arrange: the model streams the final answer as two chunks.
        chat_model = ChunkedChatModel(["fi", "nal"])
        policy = DecisionQueuePolicy([final_decision("done")])
        token_sink = RecordingTokenSink()
        observer = RecordingObserver()
        runtime = create_runtime(
            chat_model, [], policy, observer, token_sink=token_sink
        )

        # Act
        runtime.run(Prompt("question"))

        # Assert: chunks reach the sink in order, and the runtime joins them with
        # no separator into the completed model output.
        assert [chunk.value for chunk in token_sink.chunks] == ["fi", "nal"]
        assert observer.model_outputs == [(Iteration(1), MessageContent("final"))]


class TestAgentRuntimeSessionHooks:
    def test_session_start_block_aborts_before_calling_the_model(self) -> None:
        # Arrange
        chat_model = RecordingChatModel(["final"])
        policy = DecisionQueuePolicy([final_decision("done")])
        hook = ScriptedHook(session_start=Block(MessageContent("denied")))
        runtime = create_runtime(chat_model, [], policy, hooks=hook)

        # Act
        result = runtime.run(Prompt("question"))

        # Assert: the run ends with the block reason and the model is never called.
        assert result.answer == MessageContent("denied")
        assert list(result.steps) == []
        assert chat_model.requests == []
        assert hook.calls == ["session_start", "session_end"]
        # The aborting finish still threads one real run id to every hook.
        assert all(isinstance(rid, RunId) and rid.value for rid in hook.run_ids)
        assert len(set(hook.run_ids)) == 1

    def test_user_prompt_submit_injects_context_into_the_conversation(self) -> None:
        # Arrange
        chat_model = RecordingChatModel(["final"])
        policy = DecisionQueuePolicy([final_decision("done")])
        hook = ScriptedHook(user_prompt_submit=InjectContext(MessageContent("ctx")))
        runtime = create_runtime(chat_model, [], policy, hooks=hook)

        # Act
        runtime.run(Prompt("question"))

        # Assert: the injected message reaches the model on the first turn.
        first_messages = list(chat_model.requests[0].messages)
        assert ChatMessage(USER, MessageContent("ctx")) in first_messages

    def test_session_start_injects_a_system_message_into_the_conversation(
        self,
    ) -> None:
        # Arrange
        chat_model = RecordingChatModel(["final"])
        policy = DecisionQueuePolicy([final_decision("done")])
        hook = ScriptedHook(session_start=InjectContext(MessageContent("rules")))
        runtime = create_runtime(chat_model, [], policy, hooks=hook)

        # Act
        runtime.run(Prompt("question"))

        # Assert: the injected system context reaches the model on the first turn.
        first_messages = list(chat_model.requests[0].messages)
        assert ChatMessage(SYSTEM, MessageContent("rules")) in first_messages

    def test_session_end_runs_after_the_session_hooks(self) -> None:
        # Arrange
        chat_model = RecordingChatModel(["final"])
        policy = DecisionQueuePolicy([final_decision("done")])
        hook = ScriptedHook()
        runtime = create_runtime(chat_model, [], policy, hooks=hook)

        # Act
        runtime.run(Prompt("question"))

        # Assert: the final answer routes through the stop hook before the end.
        assert hook.calls == [
            "session_start",
            "user_prompt_submit",
            "stop",
            "session_end",
        ]


class TestAgentRuntimeToolHooks:
    def test_pre_tool_use_block_skips_the_tool_and_feeds_the_reason(self) -> None:
        # Arrange
        chat_model = RecordingChatModel(["tool", "final"])
        policy = DecisionQueuePolicy(
            [tool_decision("calculator", "2 + 2"), final_decision("ok")]
        )
        tool = RecordingAgentTool()
        hook = ScriptedHook(pre_tool_use=Block(MessageContent("not allowed")))
        runtime = create_runtime(chat_model, [tool], policy, hooks=hook)

        # Act
        runtime.run(Prompt("question"))

        # Assert: the tool never ran; the block reason is the failed observation.
        assert tool.requests == []
        assert policy.tool_results == [
            ToolRunResult(
                ToolName("calculator"),
                ToolOutput("not allowed"),
                succeeded=False,
            )
        ]

    def test_pre_tool_use_inject_adds_context_and_still_runs_the_tool(self) -> None:
        # Arrange
        chat_model = RecordingChatModel(["tool", "final"])
        policy = DecisionQueuePolicy(
            [tool_decision("calculator", "2 + 2"), final_decision("4")]
        )
        tool = RecordingAgentTool()
        hook = ScriptedHook(pre_tool_use=InjectContext(MessageContent("hint")))
        runtime = create_runtime(chat_model, [tool], policy, hooks=hook)

        # Act
        runtime.run(Prompt("question"))

        # Assert: the tool still ran and the injected hint reaches the next turn.
        assert tool.requests == [
            ToolRunRequest(ToolName("calculator"), ToolInput("2 + 2"))
        ]
        assert ChatMessage(USER, MessageContent("hint")) in list(
            chat_model.requests[1].messages
        )

    def test_post_tool_use_block_appends_the_reason_as_feedback(self) -> None:
        # Arrange
        chat_model = RecordingChatModel(["tool", "final"])
        policy = DecisionQueuePolicy(
            [tool_decision("calculator", "2 + 2"), final_decision("4")]
        )
        tool = RecordingAgentTool()
        hook = ScriptedHook(post_tool_use=Block(MessageContent("rejected")))
        runtime = create_runtime(chat_model, [tool], policy, hooks=hook)

        # Act
        runtime.run(Prompt("question"))

        # Assert: the tool ran, and the block reason is fed back to the model.
        assert tool.requests == [
            ToolRunRequest(ToolName("calculator"), ToolInput("2 + 2"))
        ]
        assert ChatMessage(USER, MessageContent("rejected")) in list(
            chat_model.requests[1].messages
        )

    def test_post_tool_use_injects_feedback_after_the_observation(self) -> None:
        # Arrange
        chat_model = RecordingChatModel(["tool", "final"])
        policy = DecisionQueuePolicy(
            [tool_decision("calculator", "2 + 2"), final_decision("4")]
        )
        tool = RecordingAgentTool()
        hook = ScriptedHook(post_tool_use=InjectContext(MessageContent("checked")))
        runtime = create_runtime(chat_model, [tool], policy, hooks=hook)

        # Act
        runtime.run(Prompt("question"))

        # Assert: the feedback reaches the model on the next turn.
        second_turn = list(chat_model.requests[1].messages)
        assert ChatMessage(USER, MessageContent("checked")) in second_turn

    def test_threads_run_id_iteration_and_payloads_to_every_hook(self) -> None:
        # Arrange: one tool turn then a final answer, hooks proceeding throughout.
        chat_model = RecordingChatModel(["tool", "final"])
        policy = DecisionQueuePolicy(
            [tool_decision("calculator", "2 + 2"), final_decision("4")]
        )
        hook = ScriptedHook()
        runtime = create_runtime(chat_model, [RecordingAgentTool()], policy, hooks=hook)

        # Act
        result = runtime.run(Prompt("question"))

        # Assert: each hook is called with the correct correlation id and payload.
        assert hook.calls == [
            "session_start",
            "user_prompt_submit",
            "pre_tool_use",
            "post_tool_use",
            "stop",
            "session_end",
        ]
        assert len(set(hook.run_ids)) == 1
        assert hook.prompts == [Prompt("question"), Prompt("question")]
        assert hook.iterations == [Iteration(1), Iteration(1), Iteration(2)]
        assert hook.tool_calls == [
            tool_decision("calculator", "2 + 2"),
            tool_decision("calculator", "2 + 2"),
        ]
        assert hook.tool_results == [
            ToolRunResult(ToolName("calculator"), ToolOutput("4"), succeeded=True)
        ]
        assert hook.answers == [MessageContent("4")]
        assert hook.ended_with == [result]

    def test_stop_block_keeps_looping_with_the_reason_as_guidance(self) -> None:
        # Arrange: every stop is vetoed, so two final answers still exhaust the loop.
        vetoed_turns = 2
        chat_model = RecordingChatModel(["final", "final"])
        policy = DecisionQueuePolicy(
            [final_decision("first"), final_decision("second")]
        )
        hook = ScriptedHook(stop=Block(MessageContent("keep going")))
        runtime = create_runtime(
            chat_model, [], policy, hooks=hook, max_iterations=vetoed_turns
        )

        # Act
        result = runtime.run(Prompt("question"))

        # Assert: the loop ran twice and the guidance was fed back after the veto.
        assert result.answer == FALLBACK_ANSWER
        assert len(chat_model.requests) == vetoed_turns
        assert ChatMessage(USER, MessageContent("keep going")) in list(
            chat_model.requests[1].messages
        )
        # The fallback finish threads one real run id through to session end.
        assert all(isinstance(rid, RunId) and rid.value for rid in hook.run_ids)
        assert len(set(hook.run_ids)) == 1


class TestAgentRuntimeObservability:
    def test_emits_lifecycle_events_in_order_with_payloads(self) -> None:
        # Arrange
        chat_model = RecordingChatModel(["tool", "final"])
        policy = DecisionQueuePolicy(
            [tool_decision("calculator", "2 + 2"), final_decision("4")]
        )
        observer = RecordingObserver()
        runtime = create_runtime(chat_model, [RecordingAgentTool()], policy, observer)

        # Act
        result = runtime.run(Prompt("question"))

        # Assert
        assert observer.events == [
            "run_started:question",
            "model_completed",
            "decision_parsed",
            "tool_invoked",
            "model_completed",
            "decision_parsed",
            "run_finished",
        ]
        assert observer.tool_invocations == [
            (
                Iteration(1),
                ToolRunResult(ToolName("calculator"), ToolOutput("4"), succeeded=True),
            )
        ]
        assert observer.finished == [result]
        assert observer.parsed == [
            (Iteration(1), tool_decision("calculator", "2 + 2")),
            (Iteration(2), final_decision("4")),
        ]
        assert observer.model_outputs == [
            (Iteration(1), MessageContent("tool")),
            (Iteration(2), MessageContent("final")),
        ]

    def test_correlates_every_event_with_one_run_id(self) -> None:
        # Arrange
        chat_model = RecordingChatModel(["tool", "final"])
        policy = DecisionQueuePolicy(
            [tool_decision("calculator", "2 + 2"), final_decision("4")]
        )
        observer = RecordingObserver()
        runtime = create_runtime(chat_model, [RecordingAgentTool()], policy, observer)

        # Act
        runtime.run(Prompt("question"))

        # Assert: a single run shares exactly one real correlation id everywhere.
        assert len(observer.run_ids) == len(observer.events)
        assert len(set(observer.run_ids)) == 1
        assert all(isinstance(rid, RunId) and rid.value for rid in observer.run_ids)

    def test_measures_non_negative_model_and_tool_durations(self) -> None:
        # Arrange
        chat_model = RecordingChatModel(["tool", "final"])
        policy = DecisionQueuePolicy(
            [tool_decision("calculator", "2 + 2"), final_decision("4")]
        )
        observer = RecordingObserver()
        runtime = create_runtime(chat_model, [RecordingAgentTool()], policy, observer)

        # Act
        runtime.run(Prompt("question"))

        # Assert: one measurement per model/tool event, each a small elapsed delta.
        upper_bound_seconds = 10.0
        assert len(observer.model_elapsed) == observer.events.count("model_completed")
        assert len(observer.tool_elapsed) == observer.events.count("tool_invoked")
        assert observer.model_elapsed and observer.tool_elapsed
        assert all(0.0 <= e.value < upper_bound_seconds for e in observer.model_elapsed)
        assert all(0.0 <= e.value < upper_bound_seconds for e in observer.tool_elapsed)


class TestAgentRuntimeMultiTurn:
    def test_run_turn_executes_single_turn_and_updates_history(self) -> None:
        # Arrange
        chat_model = RecordingChatModel(["final1"])
        policy = DecisionQueuePolicy([final_decision("answer1")])
        runtime = create_runtime(chat_model, [], policy)

        system_message = ChatMessage(SYSTEM, MessageContent("Tools: 0"))
        initial_history = MessageHistory().with_message(system_message)

        # Act
        result, updated_history = runtime.run_turn(Prompt("question1"), initial_history)

        # Assert
        assert result.answer == MessageContent("answer1")
        assert result.steps is not None
        expected_history = initial_history.with_message(
            ChatMessage(USER, MessageContent("question1"))
        ).with_message(ChatMessage(ASSISTANT, MessageContent("final1")))
        assert list(updated_history) == list(expected_history)
        assert len(chat_model.requests) == 1
        expected_request_messages = initial_history.with_message(
            ChatMessage(USER, MessageContent("question1"))
        )
        assert chat_model.requests[0].messages == expected_request_messages

    def test_run_turn_preserves_accumulated_history_across_turns(self) -> None:
        # Arrange
        chat_model = RecordingChatModel(["final1", "final2"])
        policy = DecisionQueuePolicy(
            [final_decision("answer1"), final_decision("answer2")]
        )
        runtime = create_runtime(chat_model, [], policy)

        system_message = ChatMessage(SYSTEM, MessageContent("Tools: 0"))
        initial_history = MessageHistory().with_message(system_message)

        # Act - Turn 1
        result1, history1 = runtime.run_turn(Prompt("question1"), initial_history)

        # Act - Turn 2
        result2, history2 = runtime.run_turn(Prompt("question2"), history1)

        # Assert
        assert result1.answer == MessageContent("answer1")
        assert result2.answer == MessageContent("answer2")

        expected_history1 = initial_history.with_message(
            ChatMessage(USER, MessageContent("question1"))
        ).with_message(ChatMessage(ASSISTANT, MessageContent("final1")))
        expected_history2 = expected_history1.with_message(
            ChatMessage(USER, MessageContent("question2"))
        ).with_message(ChatMessage(ASSISTANT, MessageContent("final2")))
        assert list(history1) == list(expected_history1)
        assert list(history2) == list(expected_history2)
        expected_request1 = initial_history.with_message(
            ChatMessage(USER, MessageContent("question1"))
        )
        expected_request2 = expected_history1.with_message(
            ChatMessage(USER, MessageContent("question2"))
        )
        assert len(chat_model.requests) == 2
        assert chat_model.requests[0].messages == expected_request1
        assert chat_model.requests[1].messages == expected_request2

    def test_run_turn_records_run_id_on_observer(self) -> None:
        chat_model = RecordingChatModel(["final"])
        policy = DecisionQueuePolicy([final_decision("done")])
        observer = RecordingObserver()
        runtime = create_runtime(chat_model, [], policy, observer=observer)

        runtime.run_turn(Prompt("q"), MessageHistory())

        for run_id in observer.run_ids:
            assert isinstance(run_id, RunId)

    def test_run_turn_returns_blocked_answer_when_session_blocked(self) -> None:
        chat_model = RecordingChatModel(["unused"])
        policy = DecisionQueuePolicy([final_decision("unused")])
        hook = ScriptedHook(session_start=Block(MessageContent("blocked-reason")))
        runtime = create_runtime(chat_model, [], policy, hooks=hook)

        result, _ = runtime.run_turn(Prompt("q"), MessageHistory())

        assert result.answer == MessageContent("blocked-reason")
        assert result.steps is not None
        assert len(hook.run_ids) >= 1
        for run_id in hook.run_ids:
            assert isinstance(run_id, RunId)
        assert len(hook.prompts) >= 1
        assert hook.prompts[0] == Prompt("q")

    def test_run_turn_returns_fallback_when_iterations_exhausted(self) -> None:
        chat_model = RecordingChatModel(["invalid"])
        policy = DecisionQueuePolicy([final_decision("unused")])
        observer = RecordingObserver()
        runtime = create_runtime(
            chat_model, [], policy, max_iterations=1, observer=observer
        )

        result, _ = runtime.run_turn(Prompt("q"), MessageHistory())

        assert result.answer == FALLBACK_ANSWER
        assert result.steps is not None
        assert isinstance(observer.run_ids[-1], RunId)

    def test_run_turn_exhausts_all_available_iterations(self) -> None:
        chat_model = RecordingChatModel(["invalid", "invalid", "invalid"])
        policy = DecisionQueuePolicy([final_decision("unused")])
        observer = RecordingObserver()
        runtime = create_runtime(
            chat_model, [], policy, max_iterations=3, observer=observer
        )

        result, _ = runtime.run_turn(Prompt("q"), MessageHistory())

        assert result.answer == FALLBACK_ANSWER
        assert isinstance(observer.run_ids[-1], RunId)
        # 3 iterations: each triggers model_completed + repair = 2 events
        # Plus run_started + run_finished = 2 more
        assert len(observer.events) >= 8

    def test_run_turn_sends_correct_iteration_to_observer(self) -> None:
        chat_model = RecordingChatModel(["final"])
        policy = DecisionQueuePolicy([final_decision("done")])
        observer = RecordingObserver()
        runtime = create_runtime(chat_model, [], policy, observer=observer)

        runtime.run_turn(Prompt("q"), MessageHistory())

        assert len(observer.model_outputs) == 1
        assert observer.model_outputs[0][0] == Iteration(1)

    def test_run_turn_injected_context_uses_state(self) -> None:
        chat_model = RecordingChatModel(["final"])
        policy = DecisionQueuePolicy([final_decision("done")])
        hook = ScriptedHook(session_start=InjectContext(MessageContent("ctx")))
        observer = RecordingObserver()
        runtime = create_runtime(chat_model, [], policy, observer=observer, hooks=hook)

        result, _ = runtime.run_turn(Prompt("q"), MessageHistory())

        assert result.answer == MessageContent("done")
        assert ChatMessage(SYSTEM, MessageContent("ctx")) in list(
            chat_model.requests[0].messages
        )
