"""Shared named fakes for entry-point plugin discovery and composition tests.

Core must be testable without any real provider/tool plugin installed, so these
fakes stand in for `importlib.metadata` entry points and a `ChatModel`.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence

import pytest
from little_harness.application.ports.chat_model import ChatCompletionRequest, ChatModel
from little_harness.domain.values.text_values import MessageContent
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
