"""Named test doubles for the application layer (no inline stubs)."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from little_harness.application.ports.chat_model import ChatCompletionRequest
from little_harness.domain.decision import AgentDecision, FinalAnswer, ToolCall
from little_harness.domain.errors import AgentProtocolError
from little_harness.domain.hook_decision import HookDecision, Proceed
from little_harness.domain.message import ChatMessage
from little_harness.domain.result import AgentResult
from little_harness.domain.tool_result import ToolRunRequest, ToolRunResult
from little_harness.domain.tool_spec import ToolInputSchema, ToolSpec
from little_harness.domain.values.numeric_values import ElapsedSeconds, Iteration
from little_harness.domain.values.role import USER
from little_harness.domain.values.text_values import (
    MessageContent,
    Prompt,
    RunId,
    ToolInput,
    ToolName,
    ToolOutput,
)


class RecordingChatModel:
    """ChatModel that streams each scripted output as one chunk per request."""

    def __init__(self, outputs: Sequence[str]) -> None:
        self.requests: list[ChatCompletionRequest] = []
        self.closed = False
        self._outputs = list(outputs)

    def complete_streaming(
        self, request: ChatCompletionRequest
    ) -> Iterator[MessageContent]:
        self.requests.append(request)
        yield MessageContent(self._outputs.pop(0))

    def close(self) -> None:
        self.closed = True


class ChunkedChatModel:
    """ChatModel that streams a fixed list of chunks for a single model turn."""

    def __init__(self, chunks: Sequence[str]) -> None:
        self.requests: list[ChatCompletionRequest] = []
        self.closed = False
        self._chunks = list(chunks)

    def complete_streaming(
        self, request: ChatCompletionRequest
    ) -> Iterator[MessageContent]:
        self.requests.append(request)
        for chunk in self._chunks:
            yield MessageContent(chunk)

    def close(self) -> None:
        self.closed = True


class RecordingTokenSink:
    """TokenSink spy that records every chunk it receives, in order."""

    def __init__(self) -> None:
        self.chunks: list[MessageContent] = []

    def emit(self, chunk: MessageContent) -> None:
        self.chunks.append(chunk)


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
    """AgentObserver spy: records the event sequence and each event's payload.

    `run_ids` collects the correlation id from every event so tests can assert a
    single run shares one id; `*_elapsed` collect the per-call measurements.
    """

    def __init__(self) -> None:
        self.events: list[str] = []
        self.run_ids: list[RunId] = []
        self.model_outputs: list[tuple[Iteration, MessageContent]] = []
        self.model_elapsed: list[ElapsedSeconds] = []
        self.parsed: list[tuple[Iteration, AgentDecision]] = []
        self.tool_invocations: list[tuple[Iteration, ToolRunResult]] = []
        self.tool_elapsed: list[ElapsedSeconds] = []
        self.repairs: list[tuple[Iteration, Exception]] = []
        self.finished: list[AgentResult] = []

    def on_run_started(self, run_id: RunId, prompt: Prompt) -> None:
        self.run_ids.append(run_id)
        self.events.append(f"run_started:{prompt.value}")

    def on_model_completed(
        self,
        run_id: RunId,
        iteration: Iteration,
        output: MessageContent,
        elapsed: ElapsedSeconds,
    ) -> None:
        self.run_ids.append(run_id)
        self.events.append("model_completed")
        self.model_outputs.append((iteration, output))
        self.model_elapsed.append(elapsed)

    def on_decision_parsed(
        self, run_id: RunId, iteration: Iteration, decision: AgentDecision
    ) -> None:
        self.run_ids.append(run_id)
        self.events.append("decision_parsed")
        self.parsed.append((iteration, decision))

    def on_tool_invoked(
        self,
        run_id: RunId,
        iteration: Iteration,
        result: ToolRunResult,
        elapsed: ElapsedSeconds,
    ) -> None:
        self.run_ids.append(run_id)
        self.events.append("tool_invoked")
        self.tool_invocations.append((iteration, result))
        self.tool_elapsed.append(elapsed)

    def on_repair(self, run_id: RunId, iteration: Iteration, error: Exception) -> None:
        self.run_ids.append(run_id)
        self.events.append("repair")
        self.repairs.append((iteration, error))

    def on_run_finished(self, run_id: RunId, result: AgentResult) -> None:
        self.run_ids.append(run_id)
        self.events.append("run_finished")
        self.finished.append(result)


class ScriptedHook:
    """LifecycleHook double: returns a configured decision per point, records calls.

    Each point defaults to `Proceed()`; pass a decision to script one point while
    leaving the rest proceeding, the way `DecisionQueuePolicy` scripts decisions.
    Every argument is recorded so tests can assert correct delegation/threading.
    """

    def __init__(
        self,
        *,
        session_start: HookDecision | None = None,
        user_prompt_submit: HookDecision | None = None,
        pre_tool_use: HookDecision | None = None,
        post_tool_use: HookDecision | None = None,
        stop: HookDecision | None = None,
    ) -> None:
        self.calls: list[str] = []
        self.run_ids: list[RunId] = []
        self.prompts: list[Prompt] = []
        self.iterations: list[Iteration] = []
        self.tool_calls: list[ToolCall] = []
        self.tool_results: list[ToolRunResult] = []
        self.answers: list[MessageContent] = []
        self.ended_with: list[AgentResult] = []
        self._session_start = session_start or Proceed()
        self._user_prompt_submit = user_prompt_submit or Proceed()
        self._pre_tool_use = pre_tool_use or Proceed()
        self._post_tool_use = post_tool_use or Proceed()
        self._stop = stop or Proceed()

    def on_session_start(self, run_id: RunId, prompt: Prompt) -> HookDecision:
        self.calls.append("session_start")
        self.run_ids.append(run_id)
        self.prompts.append(prompt)
        return self._session_start

    def on_user_prompt_submit(self, run_id: RunId, prompt: Prompt) -> HookDecision:
        self.calls.append("user_prompt_submit")
        self.run_ids.append(run_id)
        self.prompts.append(prompt)
        return self._user_prompt_submit

    def on_pre_tool_use(
        self, run_id: RunId, iteration: Iteration, call: ToolCall
    ) -> HookDecision:
        self.calls.append("pre_tool_use")
        self.run_ids.append(run_id)
        self.iterations.append(iteration)
        self.tool_calls.append(call)
        return self._pre_tool_use

    def on_post_tool_use(
        self,
        run_id: RunId,
        iteration: Iteration,
        call: ToolCall,
        result: ToolRunResult,
    ) -> HookDecision:
        self.calls.append("post_tool_use")
        self.run_ids.append(run_id)
        self.iterations.append(iteration)
        self.tool_calls.append(call)
        self.tool_results.append(result)
        return self._post_tool_use

    def on_stop(
        self, run_id: RunId, iteration: Iteration, answer: MessageContent
    ) -> HookDecision:
        self.calls.append("stop")
        self.run_ids.append(run_id)
        self.iterations.append(iteration)
        self.answers.append(answer)
        return self._stop

    def on_session_end(self, run_id: RunId, result: AgentResult) -> None:
        self.calls.append("session_end")
        self.run_ids.append(run_id)
        self.ended_with.append(result)


def final_decision(answer: str) -> FinalAnswer:
    return FinalAnswer(MessageContent(answer))


def tool_decision(name: str, tool_input: str) -> ToolCall:
    return ToolCall(ToolName(name), ToolInput(tool_input))
