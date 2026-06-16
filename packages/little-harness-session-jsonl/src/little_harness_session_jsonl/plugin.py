"""JSONL implementation of SessionPlugin."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from little_harness.application.ports.agent_observer import AgentObserver
from little_harness.application.ports.agent_policy import AgentPolicy
from little_harness.application.ports.session_plugin import SessionPlugin
from little_harness.application.ports.session_repository import SessionRepository
from little_harness.domain.values.text_values import SessionId

from little_harness_session_jsonl.infrastructure.jsonl_appender import JsonlFileAppender
from little_harness_session_jsonl.infrastructure.jsonl_observer import (
    JsonlSessionObserver,
)
from little_harness_session_jsonl.infrastructure.jsonl_repository import (
    JsonlSessionRepository,
)


class JsonlSessionPlugin(SessionPlugin):
    """Provides a JSONL-backed observer and repository for a given session."""

    def __init__(
        self,
        storage_dir: Path,
        policy: AgentPolicy,
        session_id: SessionId | None = None,
        parent_id: SessionId | None = None,
    ) -> None:
        """Initialize the plugin with storage directory, session ID, and policy."""
        self._storage_dir = storage_dir
        self._policy = policy
        self._session_id = session_id or SessionId(
            str(uuid.uuid4())  # pragma: no mutate
        )
        self._parent_id = parent_id
        self._storage_dir.mkdir(parents=True, exist_ok=True)  # pragma: no mutate

        file_path = (
            self._storage_dir / f"{self._session_id.value}.jsonl"
        )  # pragma: no mutate
        self._appender = JsonlFileAppender(file_path)

    @property
    def session_id(self) -> SessionId:
        """Get the current session ID."""
        return self._session_id

    @property
    def parent_id(self) -> SessionId | None:
        """Get the parent session ID, if this session was forked."""
        return self._parent_id

    def observer(self) -> AgentObserver:
        """Create and return a JSONL-backed observer."""
        return JsonlSessionObserver(self._session_id, self._appender, self._parent_id)

    def repository(self) -> SessionRepository:
        """Create and return a JSONL-backed repository."""
        return JsonlSessionRepository(self._storage_dir, self._policy)

    def fork(self) -> JsonlSessionPlugin:
        """Create a new session forked from this one, referencing it as parent."""
        return JsonlSessionPlugin(
            storage_dir=self._storage_dir,
            policy=self._policy,
            parent_id=self._session_id,
        )


def build_plugin(
    policy: AgentPolicy, session_id: SessionId | None = None
) -> JsonlSessionPlugin:
    """Build a JsonlSessionPlugin with defaults from environment or home dir."""
    # Read the storage path from environment or use default ~/.little-harness/sessions
    home = Path.home()  # pragma: no mutate
    default_dir = home / ".little-harness" / "sessions"  # pragma: no mutate
    storage_dir_str = os.environ.get(
        "LITTLE_HARNESS_SESSION_DIR", str(default_dir)
    )  # pragma: no mutate

    return JsonlSessionPlugin(Path(storage_dir_str), policy, session_id)
