"""Scripted provider for deterministic policy integration tests."""

from __future__ import annotations

import importlib.metadata
from collections.abc import Iterator, Mapping, Sequence

import pytest
from little_harness import plugin_discovery
from little_harness.application.ports.chat_model import ChatCompletionRequest
from little_harness.domain.values.text_values import MessageContent


class ScriptedChatModel:
    """Yields one scripted JSON output per model turn, in order."""

    def __init__(self, outputs: Sequence[str]) -> None:
        self._outputs = list(outputs)

    def complete_streaming(
        self, request: ChatCompletionRequest
    ) -> Iterator[MessageContent]:
        del request
        yield MessageContent(self._outputs.pop(0))

    def close(self) -> None:
        return None


class ScriptedProviderEntryPoint:
    """Entry point exposing the scripted provider as `scripted`."""

    name = "scripted"

    def __init__(self, outputs: Sequence[str]) -> None:
        self._outputs = outputs

    def load(self) -> object:
        def build(_options: Mapping[str, str]) -> ScriptedChatModel:
            return ScriptedChatModel(self._outputs)

        return build


@pytest.fixture
def install_scripted_provider(monkeypatch: pytest.MonkeyPatch) -> object:
    def install(outputs: Sequence[str]) -> None:
        scripted = ScriptedProviderEntryPoint(outputs)
        real_entry_points = importlib.metadata.entry_points

        def patched(*, group: str, name: str | None = None) -> Sequence[object]:
            if group == plugin_discovery.PROVIDER_GROUP:
                points: list[object] = [scripted]
            else:
                points = list(real_entry_points(group=group))
            if name is None:
                return points
            return [point for point in points if getattr(point, "name", None) == name]

        monkeypatch.setattr(plugin_discovery, "entry_points", patched)

    return install
