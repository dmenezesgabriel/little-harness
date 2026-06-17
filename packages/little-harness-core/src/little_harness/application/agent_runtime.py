"""The agent loop use case: orchestrates model, policy, tools, and observer."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from uuid import uuid4

from little_harness.application.agent_dependencies import AgentDependencies
from little_harness.application.decision_handler import (
    IterationContext,
    LoopDecisionVisitor,
    ModelRequestApplier,
    OutputReplacingApplier,
)
from little_harness.application.loop_state import AgentLoopState
from little_harness.application.ports.chat_model import ChatCompletionRequest
from little_harness.domain.decision import AgentDecision
from little_harness.domain.errors import AgentProtocolError
from little_harness.domain.hook_decision import Block, InjectContext, Proceed
from little_harness.domain.message import ChatMessage
from little_harness.domain.message_history import MessageHistory
from little_harness.domain.result import AgentResult
from little_harness.domain.skill import Skill
from little_harness.domain.step import AgentStep
from little_harness.domain.steps import AgentSteps
from little_harness.domain.values.numeric_values import (
    ElapsedSeconds,
    Iteration,
    MaxIterations,
    MaxTokens,
    RepeatPenalty,
    Temperature,
    TopP,
)
from little_harness.domain.values.role import ASSISTANT, SYSTEM, USER, Role
from little_harness.domain.values.text_values import MessageContent, Prompt, RunId

FALLBACK_ANSWER = MessageContent(
    "The agent reached the maximum number of iterations without producing a "
    "final answer."
)


@dataclass(frozen=True)
class AgentRuntimeConfig:
    """Agent loop configuration: iteration cap, temperature, sampling params."""

    max_iterations: MaxIterations
    temperature: Temperature
    max_tokens: MaxTokens
    top_p: TopP | None = None
    repeat_penalty: RepeatPenalty | None = None


class SessionDecisionApplier:
    """Applies a session hook's decision: inject a message, or report a block.

    Implements `HookDecisionVisitor[MessageContent | None]`; the returned reason
    is the answer the run aborts with, or None to continue.
    """

    def __init__(self, state: AgentLoopState, role: Role) -> None:
        """See class docstring for argument descriptions."""
        self._state = state
        self._role = role

    def visit_proceed(self, _decision: Proceed) -> MessageContent | None:
        """Continue without injecting or blocking."""
        return None

    def visit_inject_context(self, decision: InjectContext) -> MessageContent | None:
        """Inject the context message under the configured role."""
        self._state.append_message(ChatMessage(self._role, decision.content))
        return None

    def visit_block(self, decision: Block) -> MessageContent | None:
        """Abort the run with the block reason."""
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
        """See class docstring for argument descriptions."""
        self._dependencies = dependencies
        self._config = config

    def build_system_message(self) -> ChatMessage:
        """Build the system message from the policy, tool specs, and skills."""
        specs = self._dependencies.tool_registry.specs()
        prompt = self._dependencies.policy.system_prompt(specs)
        skills_content = self._format_skills_for_prompt()

        if skills_content:
            combined = f"{prompt.value}\n\n{skills_content}"
            return ChatMessage(SYSTEM, MessageContent(combined))

        return ChatMessage(SYSTEM, prompt)

    @staticmethod
    def _format_skills_content(skills: Sequence[Skill]) -> str:
        """Format loaded skills as an XML block for the system prompt."""
        if not skills:
            return ""

        lines = [
            "The following skills provide specialized instructions for specific tasks.",
            "Read the full skill file when the task matches its description.",
            "",
            "<available_skills>",
        ]

        for skill in skills:
            lines.append("  <skill>")
            lines.append(f"    <name>{skill.name.value}</name>")
            lines.append(f"    <description>{skill.description.value}</description>")
            lines.append(f"    <location>{skill.file_path}</location>")
            lines.append("  </skill>")

        lines.append("</available_skills>")
        return "\n".join(lines)

    def _format_skills_for_prompt(self) -> str:
        """Load skills and format them for the system prompt."""
        skills = self._dependencies.skill_loader.load_skills()
        return self._format_skills_content(skills)

    def run(self, prompt: Prompt) -> AgentResult:
        """Run a single-turn session from prompt to result."""
        initial_history = MessageHistory().with_message(self.build_system_message())
        result, _ = self.run_turn(prompt, initial_history)
        return result

    def run_turn(
        self, prompt: Prompt, messages: MessageHistory
    ) -> tuple[AgentResult, MessageHistory]:
        """Run a multi-turn session and return the result along with final messages."""
        run_id = RunId(uuid4().hex)
        self._dependencies.observer.on_run_started(run_id, prompt)
        started_at = time.perf_counter()
        user_message = ChatMessage(USER, MessageContent(prompt.value))
        next_messages = messages.with_message(user_message)
        state = AgentLoopState(next_messages)

        blocked = self._begin_session(run_id, prompt, state)
        if blocked is not None:
            result = self._finish(run_id, started_at, blocked, state.steps)
            return result, state.messages

        for index in range(1, self._config.max_iterations.value + 1):
            answer = self._run_iteration(run_id, state, prompt, Iteration(index))

            if answer is not None:
                result = self._finish(run_id, started_at, answer, state.steps)
                return result, state.messages

        result = self._finish(run_id, started_at, FALLBACK_ANSWER, state.steps)
        return result, state.messages

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
        start = self._dependencies.hooks.on_turn_start(run_id, iteration, prompt)
        blocked = start.accept(SessionDecisionApplier(state, SYSTEM))

        if blocked is not None:
            return blocked

        model_req = self._dependencies.hooks.on_model_request(run_id, iteration)
        fake_output = model_req.accept(ModelRequestApplier(state))

        output = fake_output
        if output is None:
            output = self._complete_timed(run_id, iteration, state.messages)
            model_resp = self._dependencies.hooks.on_model_response(
                run_id, iteration, output
            )
            output = model_resp.accept(
                OutputReplacingApplier(state, output)
            )

        state.append_message(ChatMessage(ASSISTANT, output))

        turn_end = self._dependencies.hooks.on_turn_end(
            run_id, iteration, output
        )
        output = turn_end.accept(OutputReplacingApplier(state, output))

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
            top_p=self._config.top_p,
            repeat_penalty=self._config.repeat_penalty,
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
