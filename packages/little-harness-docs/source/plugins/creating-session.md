# Creating a Session Plugin

A session plugin provides a durable way to record and load session histories in the background. It interfaces with the CLI by injecting a customized `SessionRepository` and `AgentObserver` directly into the `InteractiveConsole` initialization process.

To write a session plugin, implement the `SessionPlugin` application port. The core will discover your plugin through the `little_harness.session_plugins` entry point group.

## The Port

```python
from typing import Protocol

from little_harness.application.ports.agent_observer import AgentObserver
from little_harness.application.ports.session_repository import SessionRepository
from little_harness.presentation.cli.app_config import AppConfig

class SessionPlugin(Protocol):
    """Provides observation and persistence capabilities for interactive sessions."""
    def build_observer(
        self, session_id: str, config: AppConfig
    ) -> AgentObserver: ...

    def build_repository(
        self, config: AppConfig
    ) -> SessionRepository: ...
```

## The Implementation

To implement your plugin, you need to map agent runtime events to a durable format, and be able to rebuild `MessageHistory` chains by decoding those formats.
For example, the default JSONL file plugin uses the agent policy's serialization features to parse `ToolRunResult` observations.

```python
from pathlib import Path

from little_harness.application.ports.agent_observer import AgentObserver
from little_harness.application.ports.session_repository import SessionRepository
from little_harness.presentation.cli.app_config import AppConfig

from little_harness_session_jsonl.infrastructure.jsonl_observer import JsonlSessionObserver
from little_harness_session_jsonl.infrastructure.jsonl_repository import JsonlSessionRepository
from little_harness_session_jsonl.infrastructure.jsonl_appender import JsonlFileAppender


class JsonlSessionPlugin:
    def build_observer(
        self, session_id: str, config: AppConfig
    ) -> AgentObserver:
        storage_dir = Path.home() / ".little-harness" / "sessions"
        storage_dir.mkdir(parents=True, exist_ok=True)
        file_path = storage_dir / f"{session_id}.jsonl"
        return JsonlSessionObserver(SessionId(session_id), JsonlFileAppender(file_path))

    def build_repository(
        self, config: AppConfig
    ) -> SessionRepository:
        from little_harness.composition import discover_policy
        storage_dir = Path.home() / ".little-harness" / "sessions"
        return JsonlSessionRepository(storage_dir, discover_policy(config.policy))

def build_plugin() -> SessionPlugin:
    return JsonlSessionPlugin()
```

## Registering the Plugin

Register your implementation in `pyproject.toml` so the CLI can discover it under the `little_harness.session_plugins` namespace:

```toml
[project.entry-points."little_harness.session_plugins"]
jsonl = "little_harness_session_jsonl.plugin:build_plugin"
```

## Invoking

Users can then launch the plugin natively with:
```bash
uv run little-harness --session-plugin jsonl
```
