from __future__ import annotations

from collections.abc import Sequence

from local_llm.application.agent_dependencies import AgentDependencies
from local_llm.application.agent_runtime import (
    FALLBACK_ANSWER,
    AgentRuntime,
    AgentRuntimeConfig,
)
from local_llm.application.ports.agent_tool import AgentTool
from local_llm.application.tool_registry import ToolRegistry
from local_llm.domain.message import ChatMessage
from local_llm.domain.message_history import MessageHistory
from local_llm.domain.tool_result import ToolRunRequest, ToolRunResult
from local_llm.domain.values.numeric_values import (
    Iteration,
    MaxIterations,
    MaxTokens,
    Temperature,
)
from local_llm.domain.values.role import ASSISTANT, SYSTEM, USER
from local_llm.domain.values.text_values import (
    MessageContent,
    Prompt,
    ToolInput,
    ToolName,
    ToolOutput,
)
from tests.application.fakes import (
    DecisionQueuePolicy,
    FailingAgentTool,
    RecordingAgentTool,
    RecordingChatModel,
    RecordingObserver,
    final_decision,
    tool_decision,
)


def create_runtime(
    chat_model: RecordingChatModel,
    tools: Sequence[AgentTool],
    policy: DecisionQueuePolicy,
    observer: RecordingObserver | None = None,
    max_iterations: int = 3,
) -> AgentRuntime:
    dependencies = AgentDependencies(
        chat_model=chat_model,
        tool_registry=ToolRegistry(tools),
        policy=policy,
        observer=observer or RecordingObserver(),
    )
    config = AgentRuntimeConfig(
        max_iterations=MaxIterations(max_iterations),
        temperature=Temperature(0.0),
        max_tokens=MaxTokens(128),
    )
    return AgentRuntime(dependencies, config)


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
        runtime = create_runtime(chat_model, [], policy)

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
        assert [decision for _, decision in observer.parsed] == [
            tool_decision("calculator", "2 + 2"),
            final_decision("4"),
        ]
