from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

from local_llm.chat import ChatCompletionRequest, ChatMessage, ChatModel
from local_llm.tools import AgentTool, ToolRunRequest, ToolRunResult, ToolSpec

AgentActionKind = Literal["tool", "final"]


class AgentProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class AgentDecision:
    kind: AgentActionKind
    tool_name: str | None
    tool_input: str | None
    final_answer: str | None


@dataclass(frozen=True)
class AgentStep:
    iteration: int
    model_output: str
    decision: AgentDecision | None
    observation: str


@dataclass(frozen=True)
class AgentResult:
    answer: str
    elapsed_seconds: float
    steps: tuple[AgentStep, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class AgentRuntimeConfig:
    max_iterations: int
    temperature: float
    max_tokens: int


class AgentPolicy(Protocol):
    def system_prompt(self, tools: Sequence[ToolSpec]) -> str:
        """Return the system prompt for the available tools.

        Example:
            prompt = policy.system_prompt([calculator.spec])
        """
        ...

    def parse_model_output(self, output: str) -> AgentDecision:
        """Parse model text into a validated agent decision.

        Example:
            decision = policy.parse_model_output(model_output)
        """
        ...

    def build_tool_observation_message(
        self,
        original_prompt: str,
        tool_result: ToolRunResult,
    ) -> ChatMessage:
        """Return the next user message after a tool execution.

        Example:
            message = policy.build_tool_observation_message(prompt, result)
        """
        ...

    def build_repair_message(
        self,
        original_prompt: str,
        error: Exception,
    ) -> ChatMessage:
        """Return a user message that asks the model to repair invalid output.

        Example:
            message = policy.build_repair_message(prompt, error)
        """
        ...


class AgentRuntime:
    def __init__(
        self,
        chat_model: ChatModel,
        tools: Sequence[AgentTool],
        policy: AgentPolicy,
        config: AgentRuntimeConfig,
    ) -> None:
        self._chat_model = chat_model
        self._tools_by_name = create_tool_registry(tools)
        self._policy = policy
        self._config = config

    def run(self, prompt: str) -> AgentResult:
        started_at = time.perf_counter()
        messages = self._create_initial_messages(prompt)
        steps: list[AgentStep] = []

        for iteration in range(1, self._config.max_iterations + 1):
            model_output = self._complete(messages)
            messages.append(ChatMessage(role="assistant", content=model_output))

            try:
                decision = self._policy.parse_model_output(model_output)
            except AgentProtocolError as error:
                observation = self._repair_invalid_output(messages, prompt, error)
                steps.append(AgentStep(iteration, model_output, None, observation))
                continue

            if decision.kind == "final":
                return self._create_result(started_at, decision.final_answer, steps)

            try:
                tool_result = self._run_tool(decision)
            except AgentProtocolError as error:
                observation = self._repair_invalid_output(messages, prompt, error)
                steps.append(AgentStep(iteration, model_output, None, observation))
                continue

            messages.append(
                self._policy.build_tool_observation_message(prompt, tool_result)
            )
            steps.append(
                AgentStep(iteration, model_output, decision, tool_result.output)
            )

        return self._create_result(started_at, None, steps)

    def _create_initial_messages(self, prompt: str) -> list[ChatMessage]:
        tool_specs = [tool.spec for tool in self._tools_by_name.values()]

        return [
            ChatMessage(role="system", content=self._policy.system_prompt(tool_specs)),
            ChatMessage(role="user", content=prompt),
        ]

    def _complete(self, messages: Sequence[ChatMessage]) -> str:
        request = ChatCompletionRequest(
            messages=tuple(messages),
            temperature=self._config.temperature,
            max_tokens=self._config.max_tokens,
        )
        response = self._chat_model.complete(request)
        return response.content

    def _repair_invalid_output(
        self,
        messages: list[ChatMessage],
        prompt: str,
        error: AgentProtocolError,
    ) -> str:
        message = self._policy.build_repair_message(prompt, error)
        messages.append(message)
        return message.content

    def _run_tool(self, decision: AgentDecision) -> ToolRunResult:
        tool_name = require_tool_name(decision)
        tool_input = require_tool_input(decision)
        tool = self._tools_by_name.get(tool_name)

        if tool is None:
            return ToolRunResult(
                tool_name=tool_name,
                output=f"Unknown tool: {tool_name}. Expected one registered tool.",
                succeeded=False,
            )

        try:
            return tool.run(ToolRunRequest(tool_name, tool_input))
        except Exception as error:
            return ToolRunResult(tool_name, f"Tool error: {error}", False)

    def _create_result(
        self,
        started_at: float,
        answer: str | None,
        steps: list[AgentStep],
    ) -> AgentResult:
        elapsed_seconds = time.perf_counter() - started_at

        return AgentResult(
            answer=answer
            or (
                "The agent reached the maximum number of iterations without "
                "producing a final answer."
            ),
            elapsed_seconds=elapsed_seconds,
            steps=tuple(steps),
        )


def create_tool_registry(tools: Sequence[AgentTool]) -> dict[str, AgentTool]:
    registry: dict[str, AgentTool] = {}

    for tool in tools:
        tool_name = tool.spec.name.strip()

        if tool_name == "":
            raise ValueError("Expected non-empty tool name, got empty string.")

        if tool_name in registry:
            raise ValueError(
                f"Duplicate tool name: {tool_name}. Expected unique names."
            )

        registry[tool_name] = tool

    return registry


def require_tool_name(decision: AgentDecision) -> str:
    if decision.tool_name is None:
        raise AgentProtocolError(
            f"Tool decision tool_name is invalid: {decision.tool_name}. "
            "Expected non-null string."
        )

    return decision.tool_name.strip()


def require_tool_input(decision: AgentDecision) -> str:
    if decision.tool_input is None:
        raise AgentProtocolError(
            f"Tool decision tool_input is invalid: {decision.tool_input}. "
            "Expected non-null string."
        )

    return decision.tool_input.strip()
