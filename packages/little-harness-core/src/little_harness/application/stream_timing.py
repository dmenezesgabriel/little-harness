"""Measure a streamed completion: assemble content while timing first token.

Kept separate from the agent loop so the timing branches (first-token capture,
chunk counting) are unit-testable in isolation with an injected clock.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from little_harness.domain.values.numeric_values import ElapsedSeconds
from little_harness.domain.values.text_values import MessageContent


@dataclass(frozen=True)
class StreamMeasurement:
    """Assembled stream content plus its first-token latency and token count."""

    content: MessageContent
    time_to_first_token: ElapsedSeconds | None
    output_tokens: int


def measure_stream(
    chunks: Iterable[MessageContent],
    emit: Callable[[MessageContent], None],
    now: Callable[[], float] = time.perf_counter,
) -> StreamMeasurement:
    """Drain `chunks`, emitting each, and report first-token latency + count.

    `time_to_first_token` is the delay from entry to the first chunk; it stays
    None when the stream is empty. `output_tokens` is the chunk count, a proxy
    for tokens generated (llama.cpp streams ~one token per chunk).
    """
    start = now()
    time_to_first_token: ElapsedSeconds | None = None
    pieces: list[str] = []

    for chunk in chunks:
        if time_to_first_token is None:
            time_to_first_token = ElapsedSeconds(now() - start)
        pieces.append(chunk.value)
        emit(chunk)

    return StreamMeasurement(
        content=MessageContent("".join(pieces)),
        time_to_first_token=time_to_first_token,
        output_tokens=len(pieces),
    )
