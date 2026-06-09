# Creating a Session Plugin

A session plugin provides durable recording and loading of session histories. It implements the `SessionPlugin` port, which returns an `AgentObserver` for writing events and a `SessionRepository` for reading past history.

## The Port

```python
from typing import Protocol

from little_harness.application.ports.agent_observer import AgentObserver
from little_harness.application.ports.session_repository import SessionRepository


class SessionPlugin(Protocol):
    """Provides observation and persistence for interactive sessions."""

    def observer(self) -> AgentObserver:
        """Returns an observer that records session events."""
        ...

    def repository(self) -> SessionRepository:
        """Returns a repository that loads past session history."""
        ...
```

## The Implementation

Each session event is mapped to a durable format. The repository rebuilds `MessageHistory` chains by decoding those stored events.

```python
from pathlib import Path

from little_harness.application.ports.agent_observer import AgentObserver
from little_harness.application.ports.agent_policy import AgentPolicy
from little_harness.application.ports.session_plugin import SessionPlugin
from little_harness.application.ports.session_repository import SessionRepository
from little_harness.domain.values.text_values import SessionId


class JsonlSessionPlugin(SessionPlugin):
    def __init__(
        self,
        storage_dir: Path,
        policy: AgentPolicy,
        session_id: SessionId | None = None,
    ) -> None:
        self._storage_dir = storage_dir
        self._policy = policy
        self._session_id = session_id or SessionId(str(uuid.uuid4()))
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        file_path = self._storage_dir / f"{self._session_id.value}.jsonl"
        self._appender = JsonlFileAppender(file_path)

    @property
    def session_id(self) -> SessionId:
        return self._session_id

    def observer(self) -> AgentObserver:
        return JsonlSessionObserver(self._session_id, self._appender)

    def repository(self) -> SessionRepository:
        return JsonlSessionRepository(self._storage_dir, self._policy)


def build_plugin(
    policy: AgentPolicy, session_id: SessionId | None = None
) -> SessionPlugin:
    home = Path.home()
    default_dir = home / ".little-harness" / "sessions"
    storage_dir = Path(os.environ.get("LITTLE_HARNESS_SESSION_DIR", str(default_dir)))
    return JsonlSessionPlugin(storage_dir, policy, session_id)
```

## Register the Entry Point

Register your builder under the `little_harness.session_plugins` group:

```toml
[project.entry-points."little_harness.session_plugins"]
jsonl = "little_harness_session_jsonl.plugin:build_plugin"
```

The `SessionPlugin` port and the `SessionRepository` port are defined in core, so plugins use them at build time via `little_harness.application.ports.session_plugin` and `little_harness.application.ports.session_repository`.
