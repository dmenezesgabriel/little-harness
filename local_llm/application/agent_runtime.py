"""The agent loop use case: orchestrates model, policy, tools, and observer."""

from __future__ import annotations

import time
from dataclasses import dataclass

from local_llm.application.agent_dependencies import AgentDependencies
from local_llm.application.decision_handler import IterationContext, LoopDecisionVisitor
from local_llm.application.loop_state import AgentLoopState
from local_llm.application.ports.chat_model import ChatCompletionRequest
from local_llm.domain.decision import AgentDecision
from local_llm.domain.errors import AgentProtocolError
from local_llm.domain.message import ChatMessage
from local_llm.domain.message_history import MessageHistory
from local_llm.domain.result import AgentResult
from local_llm.domain.step import AgentStep
from local_llm.domain.steps import AgentSteps
from local_llm.domain.values.numeric_values import (
    ElapsedSeconds,
    Iteration,
    MaxIterations,
    MaxTokens,
    Temperature,
)
from local_llm.domain.values.role import ASSISTANT, SYSTEM, USER
from local_llm.domain.values.text_values import MessageContent, Prompt

FALLBACK_ANSWER = MessageContent(
    "The agent reached the maximum number of iterations without producing a "
    "final answer."
)


@dataclass(frozen=True)
class AgentRuntimeConfig:
    max_iterations: MaxIterations
    temperature: Temperature
    max_tokens: MaxTokens


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
        self._dependencies.observer.on_run_started(prompt)
        started_at = time.perf_counter()
        state = AgentLoopState(self._initial_messages(prompt))

        for index in range(1, self._config.max_iterations.value + 1):
            answer = self._run_iteration(state, prompt, Iteration(index))

            if answer is not None:
                return self._finish(started_at, answer, state.steps)

        return self._finish(started_at, FALLBACK_ANSWER, state.steps)

    def _run_iteration(
        self,
        state: AgentLoopState,
        prompt: Prompt,
        iteration: Iteration,
    ) -> MessageContent | None:
        output = self._complete(state.messages)
        self._dependencies.observer.on_model_completed(iteration, output)
        state.append_message(ChatMessage(ASSISTANT, output))

        decision = self._parse_or_repair(state, prompt, output, iteration)

        if decision is None:
            return None

        self._dependencies.observer.on_decision_parsed(iteration, decision)
        context = IterationContext(prompt, iteration, output, state)
        return decision.accept(LoopDecisionVisitor(self._dependencies, context))

    def _parse_or_repair(
        self,
        state: AgentLoopState,
        prompt: Prompt,
        output: MessageContent,
        iteration: Iteration,
    ) -> AgentDecision | None:
        try:
            return self._dependencies.policy.parse_model_output(output)
        except AgentProtocolError as error:
            self._dependencies.observer.on_repair(iteration, error)
            message = self._dependencies.policy.build_repair_message(prompt, error)
            state.record_step(AgentStep(iteration, output, None, message.content))
            state.append_message(message)
            return None

    def _initial_messages(self, prompt: Prompt) -> MessageHistory:
        specs = self._dependencies.tool_registry.specs()
        system = ChatMessage(SYSTEM, self._dependencies.policy.system_prompt(specs))
        user = ChatMessage(USER, MessageContent(prompt.value))
        return MessageHistory().with_message(system).with_message(user)

    def _complete(self, messages: MessageHistory) -> MessageContent:
        request = ChatCompletionRequest(
            messages,
            self._config.temperature,
            self._config.max_tokens,
        )
        return self._dependencies.chat_model.complete(request).content

    def _finish(
        self,
        started_at: float,
        answer: MessageContent,
        steps: AgentSteps,
    ) -> AgentResult:
        elapsed = ElapsedSeconds(time.perf_counter() - started_at)
        result = AgentResult(answer, elapsed, steps)
        self._dependencies.observer.on_run_finished(result)
        return result
