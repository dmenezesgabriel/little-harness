# ruff: noqa: D100, D101, D102, D103
# pyright: reportPrivateUsage=false
from pathlib import Path

from typing import Any

import pytest
from little_harness.application.ports.agent_policy import AgentPolicy
from little_harness.domain.decision import AgentDecision
from little_harness.domain.message import ChatMessage
from little_harness.domain.tool_result import ToolRunResult
from little_harness.domain.values.text_values import MessageContent, Prompt, SessionId

from little_harness_session_jsonl.infrastructure.jsonl_observer import (
    JsonlSessionObserver,
)
from little_harness_session_jsonl.infrastructure.jsonl_repository import (
    JsonlSessionRepository,
)
from little_harness_session_jsonl.plugin import JsonlSessionPlugin, build_plugin


class FakeAgentPolicy(AgentPolicy):
    def build_tool_observation_message(
        self, original_prompt: Prompt, tool_result: ToolRunResult
    ) -> ChatMessage:
        raise NotImplementedError

    def build_repair_message(
        self, original_prompt: Prompt, error: Exception
    ) -> ChatMessage:
        raise NotImplementedError

    def system_prompt(self, tools: object) -> MessageContent:
        raise NotImplementedError

    def response_schema(self, tools: object) -> Any | None:
        raise NotImplementedError

    def parse_model_output(self, output: MessageContent) -> AgentDecision: ...


def test_build_plugin_uses_env_var(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("LITTLE_HARNESS_SESSION_DIR", str(tmp_path))
    policy = FakeAgentPolicy()

    plugin = build_plugin(policy)

    assert isinstance(plugin, JsonlSessionPlugin)
    assert plugin.session_id is not None
    assert plugin._storage_dir == tmp_path
    assert plugin._policy is policy

    observer = plugin.observer()
    assert isinstance(observer, JsonlSessionObserver)
    assert observer._session_id == plugin.session_id
    assert observer._appender._file_path.parent == tmp_path

    repository = plugin.repository()
    assert isinstance(repository, JsonlSessionRepository)
    assert repository._storage_dir == tmp_path
    assert repository._policy is policy


def test_build_plugin_uses_default_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LITTLE_HARNESS_SESSION_DIR", raising=False)
    policy = FakeAgentPolicy()

    plugin = build_plugin(policy, SessionId("test-session"))

    assert plugin.session_id == SessionId("test-session")

    home = Path.home()
    expected_dir = home / ".little-harness" / "sessions"
    assert plugin._storage_dir == expected_dir

    observer = plugin.observer()
    assert isinstance(observer, JsonlSessionObserver)

    repository = plugin.repository()
    assert isinstance(repository, JsonlSessionRepository)
