"""Shared named fakes for entry-point plugin discovery and composition tests.

Core must be testable without any real provider/tool plugin installed, so these
fakes stand in for `importlib.metadata` entry points and a `ChatModel`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence

import pytest
from little_harness.application.ports.agent_observer import AgentObserver
from little_harness.application.ports.agent_policy import AgentPolicy
from little_harness.application.ports.chat_model import (
    ChatCompletionRequest,
    ChatModel,
    ResponseSchema,
)
from little_harness.domain.decision import AgentDecision, FinalAnswer
from little_harness.domain.message import ChatMessage
from little_harness.domain.tool_result import ToolRunResult
from little_harness.domain.tool_spec import ToolSpec
from little_harness.domain.values.role import USER
from little_harness.domain.values.text_values import MessageContent, Prompt
from little_harness.plugin_discovery import ChatModelBuilder

EntryPointRegistry = dict[str, list["FakeEntryPoint"]]


class FakeEntryPoint:
    """Named stand-in for importlib.metadata.EntryPoint."""

    def __init__(self, name: str, target: object) -> None:
        self.name = name
        self._target = target

    def load(self) -> object:
        return self._target


def install_entry_points(
    monkeypatch: pytest.MonkeyPatch, registry: EntryPointRegistry
) -> None:
    def query(*, group: str, name: str | None = None) -> Sequence[FakeEntryPoint]:
        points = registry.get(group, [])
        if name is None:
            return points
        return [point for point in points if point.name == name]

    monkeypatch.setattr("little_harness.plugin_discovery.entry_points", query)


class FakeChatModel:
    """Named ChatModel double that streams preset content and records close()."""

    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.requests: list[ChatCompletionRequest] = []
        self.closed = False

    def complete_streaming(
        self, request: ChatCompletionRequest
    ) -> Iterator[MessageContent]:
        self.requests.append(request)
        yield MessageContent(self._reply)

    def close(self) -> None:
        self.closed = True


def make_provider_builder(reply: str) -> ChatModelBuilder:
    def build(_options: Mapping[str, str]) -> ChatModel:
        return FakeChatModel(reply)

    return build


class FakeAgentPolicy:
    """AgentPolicy double that treats the whole model output as the final answer.

    This keeps composition tests free of any real reasoning protocol: a run ends
    after the first model reply, whatever its text.
    """

    def system_prompt(self, tools: Sequence[ToolSpec]) -> MessageContent:
        return MessageContent(f"fake system prompt ({len(tools)} tools)")

    def response_schema(self, tools: Sequence[ToolSpec]) -> ResponseSchema | None:
        return ResponseSchema({"tools": len(tools)})

    def parse_model_output(self, output: MessageContent) -> AgentDecision:
        return FinalAnswer(output)

    def build_tool_observation_message(
        self, original_prompt: Prompt, tool_result: ToolRunResult
    ) -> ChatMessage:
        observation = f"{original_prompt.value}: {tool_result.output.value}"
        return ChatMessage(USER, MessageContent(observation))

    def build_repair_message(
        self, original_prompt: Prompt, error: Exception
    ) -> ChatMessage:
        return ChatMessage(USER, MessageContent(f"{original_prompt.value}: {error}"))


def make_policy_builder() -> Callable[[], AgentPolicy]:
    def build() -> AgentPolicy:
        return FakeAgentPolicy()

    return build


class FakeObserver:
    """AgentObserver double used to assert observer discovery wiring."""

    def on_run_started(self, *_args: object, **_kwargs: object) -> None: ...

    def on_model_completed(self, *_args: object, **_kwargs: object) -> None: ...

    def on_decision_parsed(self, *_args: object, **_kwargs: object) -> None: ...

    def on_tool_invoked(self, *_args: object, **_kwargs: object) -> None: ...

    def on_repair(self, *_args: object, **_kwargs: object) -> None: ...

    def on_run_finished(self, *_args: object, **_kwargs: object) -> None: ...


def make_observer_builder() -> Callable[[], AgentObserver]:
    def build() -> AgentObserver:
        return FakeObserver()

    return build
