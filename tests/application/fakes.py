"""Named test doubles for the application layer (no inline stubs)."""

from __future__ import annotations

from collections.abc import Sequence

from local_llm.application.ports.chat_model import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from local_llm.domain.decision import AgentDecision, FinalAnswer, ToolCall
from local_llm.domain.errors import AgentProtocolError
from local_llm.domain.message import ChatMessage
from local_llm.domain.result import AgentResult
from local_llm.domain.tool_result import ToolRunRequest, ToolRunResult
from local_llm.domain.tool_spec import ToolInputSchema, ToolSpec
from local_llm.domain.values.numeric_values import Iteration
from local_llm.domain.values.role import USER
from local_llm.domain.values.text_values import (
    MessageContent,
    Prompt,
    ToolInput,
    ToolName,
    ToolOutput,
)


class RecordingChatModel:
    """ChatModel that replays scripted outputs and records each request."""

    def __init__(self, outputs: Sequence[str]) -> None:
        self.requests: list[ChatCompletionRequest] = []
        self._outputs = list(outputs)

    def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        self.requests.append(request)
        return ChatCompletionResponse(MessageContent(self._outputs.pop(0)))


class RecordingAgentTool:
    """AgentTool that always succeeds with "4" and records its requests."""

    def __init__(self, name: str = "calculator") -> None:
        self.requests: list[ToolRunRequest] = []
        self._spec = ToolSpec(
            ToolName(name),
            "Evaluate arithmetic.",
            ToolInputSchema("Arithmetic expression."),
        )

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    def run(self, request: ToolRunRequest) -> ToolRunResult:
        self.requests.append(request)
        return ToolRunResult(request.tool_name, ToolOutput("4"), succeeded=True)


class FailingAgentTool:
    """AgentTool that raises, to exercise the runtime's failure handling."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            ToolName("calculator"),
            "Fail deliberately.",
            ToolInputSchema("Arithmetic expression."),
        )

    def run(self, request: ToolRunRequest) -> ToolRunResult:
        raise RuntimeError(f"Forced failure for {request.raw_input.value}")


class DecisionQueuePolicy:
    """AgentPolicy that dequeues scripted decisions and records its inputs."""

    def __init__(self, decisions: Sequence[AgentDecision]) -> None:
        self.repair_errors: list[Exception] = []
        self.tool_results: list[ToolRunResult] = []
        self._decisions = list(decisions)

    def system_prompt(self, tools: Sequence[ToolSpec]) -> MessageContent:
        return MessageContent(f"Tools: {len(tools)}")

    def parse_model_output(self, output: MessageContent) -> AgentDecision:
        if output.value == "invalid":
            raise AgentProtocolError("Invalid model output.")

        return self._decisions.pop(0)

    def build_tool_observation_message(
        self,
        original_prompt: Prompt,
        tool_result: ToolRunResult,
    ) -> ChatMessage:
        self.tool_results.append(tool_result)
        content = f"{original_prompt.value}: {tool_result.output.value}"
        return ChatMessage(USER, MessageContent(content))

    def build_repair_message(
        self,
        original_prompt: Prompt,
        error: Exception,
    ) -> ChatMessage:
        self.repair_errors.append(error)
        return ChatMessage(
            USER, MessageContent(f"Repair {original_prompt.value}: {error}")
        )


class RecordingObserver:
    """AgentObserver spy: records the event sequence and each event's payload."""

    def __init__(self) -> None:
        self.events: list[str] = []
        self.model_outputs: list[tuple[Iteration, MessageContent]] = []
        self.parsed: list[tuple[Iteration, AgentDecision]] = []
        self.tool_invocations: list[tuple[Iteration, ToolRunResult]] = []
        self.repairs: list[tuple[Iteration, Exception]] = []
        self.finished: list[AgentResult] = []

    def on_run_started(self, prompt: Prompt) -> None:
        self.events.append(f"run_started:{prompt.value}")

    def on_model_completed(self, iteration: Iteration, output: MessageContent) -> None:
        self.events.append("model_completed")
        self.model_outputs.append((iteration, output))

    def on_decision_parsed(self, iteration: Iteration, decision: AgentDecision) -> None:
        self.events.append("decision_parsed")
        self.parsed.append((iteration, decision))

    def on_tool_invoked(self, iteration: Iteration, result: ToolRunResult) -> None:
        self.events.append("tool_invoked")
        self.tool_invocations.append((iteration, result))

    def on_repair(self, iteration: Iteration, error: Exception) -> None:
        self.events.append("repair")
        self.repairs.append((iteration, error))

    def on_run_finished(self, result: AgentResult) -> None:
        self.events.append("run_finished")
        self.finished.append(result)


def final_decision(answer: str) -> FinalAnswer:
    return FinalAnswer(MessageContent(answer))


def tool_decision(name: str, tool_input: str) -> ToolCall:
    return ToolCall(ToolName(name), ToolInput(tool_input))
