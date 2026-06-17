"""Port for loading durable session history."""

from __future__ import annotations

from typing import Protocol

from little_harness.domain.message_history import MessageHistory
from little_harness.domain.values.text_values import SessionId


class SessionRepository(Protocol):
    """Provides access to historical session states.

    Implementations restore the message history of a session from a durable backend
    (like a JSONL file, database, etc).
    """

    def load(self, session_id: SessionId) -> MessageHistory:
        """Restores a session's message history from persistent storage.

        Returns an empty MessageHistory if the session is not found or is new.
        """
        ...
