from __future__ import annotations

from collections.abc import Sequence

from local_llm.agent import (
    AgentDecision,
    AgentPolicy,
    AgentProtocolError,
    AgentRuntime,
    AgentRuntimeConfig,
)
from local_llm.chat import ChatCompletionRequest, ChatCompletionResponse, ChatMessage
from local_llm.tools import (
    AgentTool,
    ToolInputSchema,
    ToolRunRequest,
    ToolRunResult,
    ToolSpec,
)


class RecordingChatModel:
    def __init__(self, outputs: Sequence[str]) -> None:
        self.requests: list[ChatCompletionRequest] = []
        self._outputs = list(outputs)

    def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        self.requests.append(request)
        return ChatCompletionResponse(self._outputs.pop(0))


class RecordingAgentTool:
    def __init__(self, name: str = "calculator") -> None:
        self.requests: list[ToolRunRequest] = []
        self._spec = ToolSpec(
            name=name,
            description="Evaluate arithmetic.",
            input_schema=ToolInputSchema("Arithmetic expression."),
        )

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    def run(self, request: ToolRunRequest) -> ToolRunResult:
        self.requests.append(request)
        return ToolRunResult(request.tool_name, "4", True)


class DecisionQueuePolicy:
    def __init__(self, decisions: Sequence[AgentDecision]) -> None:
        self.repair_errors: list[Exception] = []
        self.tool_results: list[ToolRunResult] = []
        self._decisions = list(decisions)

    def system_prompt(self, tools: Sequence[ToolSpec]) -> str:
        return f"Tools: {len(tools)}"

    def parse_model_output(self, output: str) -> AgentDecision:
        if output == "invalid":
            raise AgentProtocolError("Invalid model output.")

        return self._decisions.pop(0)

    def build_tool_observation_message(
        self,
        original_prompt: str,
        tool_result: ToolRunResult,
    ) -> ChatMessage:
        self.tool_results.append(tool_result)
        return ChatMessage("user", f"{original_prompt}: {tool_result.output}")

    def build_repair_message(
        self,
        original_prompt: str,
        error: Exception,
    ) -> ChatMessage:
        self.repair_errors.append(error)
        return ChatMessage("user", f"Repair {original_prompt}: {error}")


class FailingAgentTool:
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="calculator",
            description="Fail deliberately.",
            input_schema=ToolInputSchema("Arithmetic expression."),
        )

    def run(self, request: ToolRunRequest) -> ToolRunResult:
        raise RuntimeError(f"Forced failure for {request.raw_input}")


def test_runtime_returns_final_answer_without_using_tools() -> None:
    # Arrange
    chat_model = RecordingChatModel(["final"])
    policy = DecisionQueuePolicy([final_decision("done")])
    tool = RecordingAgentTool()
    runtime = create_runtime(chat_model, [tool], policy)

    # Act
    result = runtime.run("question")

    # Assert
    assert result.answer == "done"
    assert result.steps == ()
    assert tool.requests == []
    assert len(chat_model.requests) == 1


def test_runtime_executes_registered_tool_and_returns_final_answer() -> None:
    # Arrange
    chat_model = RecordingChatModel(["tool", "final"])
    policy = DecisionQueuePolicy(
        [tool_decision("calculator", "2 + 2"), final_decision("4")]
    )
    tool = RecordingAgentTool()
    runtime = create_runtime(chat_model, [tool], policy)

    # Act
    result = runtime.run("question")

    # Assert
    assert result.answer == "4"
    assert len(result.steps) == 1
    assert tool.requests == [ToolRunRequest("calculator", "2 + 2")]
    assert policy.tool_results == [ToolRunResult("calculator", "4", True)]


def test_runtime_reports_unknown_tool_as_failed_observation() -> None:
    # Arrange
    chat_model = RecordingChatModel(["tool", "final"])
    policy = DecisionQueuePolicy(
        [tool_decision("missing", "2 + 2"), final_decision("fallback")]
    )
    runtime = create_runtime(chat_model, [], policy)

    # Act
    result = runtime.run("question")

    # Assert
    assert result.answer == "fallback"
    assert policy.tool_results == [
        ToolRunResult(
            "missing",
            "Unknown tool: missing. Expected one registered tool.",
            False,
        )
    ]


def test_runtime_converts_tool_exception_to_failed_observation() -> None:
    # Arrange
    chat_model = RecordingChatModel(["tool", "final"])
    policy = DecisionQueuePolicy(
        [tool_decision("calculator", "2 + 2"), final_decision("fallback")]
    )
    runtime = create_runtime(chat_model, [FailingAgentTool()], policy)

    # Act
    result = runtime.run("question")

    # Assert
    assert result.answer == "fallback"
    assert policy.tool_results == [
        ToolRunResult("calculator", "Tool error: Forced failure for 2 + 2", False)
    ]


def test_runtime_repairs_invalid_model_output() -> None:
    # Arrange
    chat_model = RecordingChatModel(["invalid", "final"])
    policy = DecisionQueuePolicy([final_decision("done")])
    runtime = create_runtime(chat_model, [], policy)

    # Act
    result = runtime.run("question")

    # Assert
    assert result.answer == "done"
    assert len(result.steps) == 1
    assert len(policy.repair_errors) == 1
    assert result.steps[0].decision is None


def test_runtime_repairs_malformed_tool_decision() -> None:
    # Arrange
    chat_model = RecordingChatModel(["tool", "final"])
    policy = DecisionQueuePolicy(
        [AgentDecision("tool", None, "2 + 2", None), final_decision("done")]
    )
    runtime = create_runtime(chat_model, [RecordingAgentTool()], policy)

    # Act
    result = runtime.run("question")

    # Assert
    assert result.answer == "done"
    assert len(result.steps) == 1
    assert len(policy.repair_errors) == 1
    assert result.steps[0].decision is None


def test_runtime_returns_fallback_after_max_iterations() -> None:
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
    result = runtime.run("question")

    # Assert
    assert result.answer == (
        "The agent reached the maximum number of iterations without producing a "
        "final answer."
    )
    assert len(result.steps) == 1


def create_runtime(
    chat_model: RecordingChatModel,
    tools: Sequence[AgentTool],
    policy: AgentPolicy,
    max_iterations: int = 3,
) -> AgentRuntime:
    return AgentRuntime(
        chat_model=chat_model,
        tools=tools,
        policy=policy,
        config=AgentRuntimeConfig(
            max_iterations=max_iterations,
            temperature=0.0,
            max_tokens=128,
        ),
    )


def final_decision(answer: str) -> AgentDecision:
    return AgentDecision("final", None, None, answer)


def tool_decision(tool_name: str, tool_input: str) -> AgentDecision:
    return AgentDecision("tool", tool_name, tool_input, None)
