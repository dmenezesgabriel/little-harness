"""CLI `TokenSink` implementations: discard tokens or stream them to a stream."""

from __future__ import annotations

import sys
from typing import TextIO

from little_harness.domain.values.text_values import MessageContent


class NullTokenSink:
    """Discards every chunk — the default when streaming is not requested."""

    def emit(self, chunk: MessageContent) -> None:
        del chunk


class StdoutTokenSink:
    """Writes each chunk to a text stream and flushes, for live CLI output.

    Example:
        StdoutTokenSink().emit(MessageContent("Hello"))
    """

    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream if stream is not None else sys.stdout

    def emit(self, chunk: MessageContent) -> None:
        self._stream.write(chunk.value)
        self._stream.flush()
