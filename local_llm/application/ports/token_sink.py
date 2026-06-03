"""Port for streaming generated tokens to the UI as they are produced."""

from __future__ import annotations

from typing import Protocol

from local_llm.domain.values.text_values import MessageContent


class TokenSink(Protocol):
    def emit(self, chunk: MessageContent) -> None:
        """Surface one generated token chunk to the user.

        Separate from `AgentObserver` (which is for logging/metrics/tracing):
        this seam is for live, user-facing output.

        Example:
            token_sink.emit(MessageContent("Hello"))
        """
        ...
