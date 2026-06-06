"""Runs the logging observer through the real core with a scripted model."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence

import pytest
from little_harness.composition import run_cli

pytestmark = pytest.mark.integration
Install = Callable[[Sequence[str]], None]


class RecordingLogger:
    """StructuredLogger fake that records each event and its fields."""

    def __init__(self) -> None:
        self.records: list[tuple[str, Mapping[str, object]]] = []

    def log(self, event: str, fields: Mapping[str, object]) -> None:
        self.records.append((event, fields))


def test_logging_observer_runs_through_the_agent_core(
    install_scripted_provider: Install,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    logger = RecordingLogger()
    install_scripted_provider([json.dumps({"action": "final", "answer": "done"})])

    def create_logger(_name: str) -> RecordingLogger:
        return logger

    monkeypatch.setattr(
        "little_harness_logging.provider.create_structured_logger",
        create_logger,
    )

    # Act
    output = run_cli(
        [
            "--provider",
            "scripted",
            "--observer",
            "logging",
            "--prompt",
            "finish",
            "--max-iterations",
            "1",
        ]
    )

    # Assert
    assert "done" in output
    assert [event for event, _fields in logger.records] == [
        "run_started",
        "model_completed",
        "decision_parsed",
        "run_finished",
    ]
