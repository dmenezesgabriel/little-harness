"""Port for session plugins that provide both observing and loading capabilities."""

from __future__ import annotations

from typing import Protocol

from little_harness.application.ports.agent_observer import AgentObserver
from little_harness.application.ports.session_repository import SessionRepository


class SessionPlugin(Protocol):
    """A plugin that manages durable session state.

    Provides both an observer to record new events during a session, and a
    repository to load past message history for an existing session.
    """

    def observer(self) -> AgentObserver:
        """Returns the observer that records session events."""
        ...

    def repository(self) -> SessionRepository:
        """Returns the repository that loads past session history."""
        ...
