"""The agent loop use case: orchestrates model, policy, tools, and observer."""

from __future__ import annotations

import time
from dataclasses import dataclass
from uuid import uuid4

from little_harness.application.agent_dependencies import AgentDependencies
from little_harness.application.decision_handler import (
    IterationContext,
    LoopDecisionVisitor,
)
from little_harness.application.loop_state import AgentLoopState
from little_harness.application.ports.chat_model import ChatCompletionRequest
from little_harness.domain.decision import AgentDecision
from little_harness.domain.errors import AgentProtocolError
from little_harness.domain.hook_decision import Block, InjectContext, Proceed
from little_harness.domain.message import ChatMessage
from little_harness.domain.message_history import MessageHistory
from little_harness.domain.result import AgentResult
from little_harness.domain.step import AgentStep
from little_harness.domain.steps import AgentSteps
from little_harness.domain.values.numeric_values import (
    ElapsedSeconds,
    Iteration,
    MaxIterations,
    MaxTokens,
    Temperature,
)
from little_harness.domain.values.role import ASSISTANT, SYSTEM, USER, Role
from little_harness.domain.values.text_values import MessageContent, Prompt, RunId

FALLBACK_ANSWER = MessageContent(
    "The agent reached the maximum number of iterations without producing a "
    "final answer."
)


@dataclass(frozen=True)
class AgentRuntimeConfig:
    max_iterations: MaxIterations
    temperature: Temperature
    max_tokens: MaxTokens


class SessionDecisionApplier:
    """Applies a session hook's decision: inject a message, or report a block.

    Implements `HookDecisionVisitor[MessageContent | None]`; the returned reason
    is the answer the run aborts with, or None to continue.
    """

    def __init__(self, state: AgentLoopState, role: Role) -> None:
        self._state = state
        self._role = role

    def visit_proceed(self, _decision: Proceed) -> MessageContent | None:
        return None

    def visit_inject_context(self, decision: InjectContext) -> MessageContent | None:
        self._state.append_message(ChatMessage(self._role, decision.content))
        return None

    def visit_block(self, decision: Block) -> MessageContent | None:
        return decision.reason


class AgentRuntime:
    """Runs the bounded reason-act loop until a final answer or the iteration cap.

    Example:
        result = AgentRuntime(dependencies, config).run(Prompt("2 + 2?"))
    """

    def __init__(
        self,
        dependencies: AgentDependencies,
        config: AgentRuntimeConfig,
    ) -> None:
        self._dependencies = dependencies
        self._config = config

    def run(self, prompt: Prompt) -> AgentResult:
        run_id = RunId(uuid4().hex)
        self._dependencies.observer.on_run_started(run_id, prompt)
        started_at = time.perf_counter()
        state = AgentLoopState(self._initial_messages(prompt))

        blocked = self._begin_session(run_id, prompt, state)
        if blocked is not None:
            return self._finish(run_id, started_at, blocked, state.steps)

        for index in range(1, self._config.max_iterations.value + 1):
            answer = self._run_iteration(run_id, state, prompt, Iteration(index))

            if answer is not None:
                return self._finish(run_id, started_at, answer, state.steps)

        return self._finish(run_id, started_at, FALLBACK_ANSWER, state.steps)

    def _begin_session(
        self, run_id: RunId, prompt: Prompt, state: AgentLoopState
    ) -> MessageContent | None:
        """Run session-start then prompt-submit hooks; a block returns its reason."""
        start = self._dependencies.hooks.on_session_start(run_id, prompt)
        blocked = start.accept(SessionDecisionApplier(state, SYSTEM))

        if blocked is not None:
            return blocked

        submit = self._dependencies.hooks.on_user_prompt_submit(run_id, prompt)
        return submit.accept(SessionDecisionApplier(state, USER))

    def _run_iteration(
        self,
        run_id: RunId,
        state: AgentLoopState,
        prompt: Prompt,
        iteration: Iteration,
    ) -> MessageContent | None:
        output = self._complete_timed(run_id, iteration, state.messages)
        state.append_message(ChatMessage(ASSISTANT, output))

        decision = self._parse_or_repair(run_id, state, prompt, output, iteration)

        if decision is None:
            return None

        self._dependencies.observer.on_decision_parsed(run_id, iteration, decision)
        context = IterationContext(run_id, prompt, iteration, output, state)
        return decision.accept(LoopDecisionVisitor(self._dependencies, context))

    def _parse_or_repair(
        self,
        run_id: RunId,
        state: AgentLoopState,
        prompt: Prompt,
        output: MessageContent,
        iteration: Iteration,
    ) -> AgentDecision | None:
        try:
            return self._dependencies.policy.parse_model_output(output)
        except AgentProtocolError as error:
            self._dependencies.observer.on_repair(run_id, iteration, error)
            message = self._dependencies.policy.build_repair_message(prompt, error)
            state.record_step(AgentStep(iteration, output, None, message.content))
            state.append_message(message)
            return None

    def _initial_messages(self, prompt: Prompt) -> MessageHistory:
        specs = self._dependencies.tool_registry.specs()
        system = ChatMessage(SYSTEM, self._dependencies.policy.system_prompt(specs))
        user = ChatMessage(USER, MessageContent(prompt.value))
        return MessageHistory().with_message(system).with_message(user)

    def _complete_timed(
        self,
        run_id: RunId,
        iteration: Iteration,
        messages: MessageHistory,
    ) -> MessageContent:
        started_at = time.perf_counter()
        output = self._complete(messages)
        elapsed = ElapsedSeconds(time.perf_counter() - started_at)
        self._dependencies.observer.on_model_completed(
            run_id, iteration, output, elapsed
        )
        return output

    def _complete(self, messages: MessageHistory) -> MessageContent:
        specs = self._dependencies.tool_registry.specs()
        request = ChatCompletionRequest(
            messages,
            self._config.temperature,
            self._config.max_tokens,
            self._dependencies.policy.response_schema(specs),
        )
        chunks: list[str] = []
        for chunk in self._dependencies.chat_model.complete_streaming(request):
            chunks.append(chunk.value)
            self._dependencies.token_sink.emit(chunk)
        return MessageContent("".join(chunks))

    def _finish(
        self,
        run_id: RunId,
        started_at: float,
        answer: MessageContent,
        steps: AgentSteps,
    ) -> AgentResult:
        elapsed = ElapsedSeconds(time.perf_counter() - started_at)
        result = AgentResult(answer, elapsed, steps)
        self._dependencies.observer.on_run_finished(run_id, result)
        self._dependencies.hooks.on_session_end(run_id, result)
        return result
