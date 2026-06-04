"""Named test doubles for the LiteLLM adapter (no real network calls)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any


class FakeDelta:
    def __init__(self, content: object) -> None:
        self.content = content


class FakeChoice:
    def __init__(self, delta: object) -> None:
        self.delta = delta


class FakeChunk:
    def __init__(self, choices: list[object]) -> None:
        self.choices = choices


class NoChoices:
    """Chunk-like object missing the `choices` attribute entirely."""


class ChoiceWithoutDelta:
    """Choice-like object missing the `delta` attribute."""


class DeltaWithoutContent:
    """Delta-like object missing the `content` attribute."""


def content_chunk(content: object) -> FakeChunk:
    return FakeChunk([FakeChoice(FakeDelta(content))])


def empty_chunk() -> FakeChunk:
    return FakeChunk([])


def chunk_with_choice(choice: object) -> FakeChunk:
    return FakeChunk([choice])


class RecordingCompletion:
    """Stand-in for litellm.completion that records kwargs and replays chunks."""

    def __init__(self, chunks: list[FakeChunk]) -> None:
        self.kwargs: dict[str, Any] = {}
        self._chunks = chunks

    def __call__(self, **kwargs: Any) -> Iterator[FakeChunk]:
        self.kwargs = kwargs
        return iter(self._chunks)


class NonStreamingCompletion:
    """Stand-in whose completion returns a non-iterator response (a dict)."""

    def __call__(self, **_kwargs: Any) -> dict[str, Any]:
        return {"choices": [{"message": {"content": "whole answer"}}]}
