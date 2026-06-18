"""Tests for measure_stream: TTFT and chunk-count timing over a token stream."""

from __future__ import annotations

from collections.abc import Iterator

from little_harness.application.stream_timing import measure_stream
from little_harness.domain.values.text_values import MessageContent


class FakeClock:
    """Monotonic clock returning each scripted reading on successive calls."""

    def __init__(self, readings: list[float]) -> None:
        self._readings = list(readings)

    def __call__(self) -> float:
        return self._readings.pop(0)


def _chunks(*values: str) -> Iterator[MessageContent]:
    return iter(MessageContent(value) for value in values)


class TestMeasureStream:
    def test_joins_chunks_and_counts_them_as_tokens(self) -> None:
        emitted: list[MessageContent] = []
        # start, then one reading per chunk for the first-token timestamp.
        clock = FakeClock([0.0, 0.2, 0.3, 0.4])

        result = measure_stream(_chunks("Hel", "lo", "!"), emitted.append, now=clock)

        assert result.content == MessageContent("Hello!")
        assert result.output_tokens == 3
        assert [chunk.value for chunk in emitted] == ["Hel", "lo", "!"]

    def test_time_to_first_token_is_first_chunk_delta(self) -> None:
        clock = FakeClock([1.0, 1.25, 1.5])

        result = measure_stream(_chunks("a", "b"), lambda _chunk: None, now=clock)

        assert result.time_to_first_token is not None
        assert result.time_to_first_token.value == 0.25

    def test_empty_stream_has_no_first_token_and_zero_tokens(self) -> None:
        clock = FakeClock([5.0])

        result = measure_stream(_chunks(), lambda _chunk: None, now=clock)

        assert result.content == MessageContent("")
        assert result.time_to_first_token is None
        assert result.output_tokens == 0
