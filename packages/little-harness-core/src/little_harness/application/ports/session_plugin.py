"""Port for session plugins that provide both observing and loading capabilities."""

from __future__ import annotations

from typing import Protocol

from little_harness.application.ports.agent_observer import AgentObserver
from little_harness.application.ports.session_repository import SessionRepository
from little_harness.domain.values.text_values import SessionId


class SessionPlugin(Protocol):
    """A plugin that manages durable session state.

    Provides both an observer to record new events during a session, and a
    repository to load past message history for an existing session.
    """

    @property
    def session_id(self) -> SessionId:
        """Return the current session ID."""
        ...

    def observer(self) -> AgentObserver:
        """Return the observer that records session events."""
        ...

    def repository(self) -> SessionRepository:
        """Return the repository that loads past session history."""
        ...

    def fork(self) -> SessionPlugin:
        """Create a new session that is a fork of this one.

        The new session shares the same storage backend and policy but has a
        fresh session ID and references this session as its parent.
        """
        ...
