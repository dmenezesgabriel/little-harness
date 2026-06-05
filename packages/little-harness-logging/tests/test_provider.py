from __future__ import annotations

import logging

import pytest
from little_harness.application.ports.agent_observer import AgentObserver
from little_harness.domain.values.text_values import Prompt, RunId
from little_harness_logging.provider import LOGGER_NAME, build


class TestBuild:
    def test_returns_an_observer_conforming_to_the_port(self) -> None:
        # Arrange / Act: the annotation forces a structural-conformance check.
        observer: AgentObserver = build()

        # Assert
        assert observer.on_run_started(RunId("rid"), Prompt("q")) is None

    def test_emits_a_json_record_under_the_agent_logger(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Arrange
        observer = build()

        # Act
        with caplog.at_level(logging.INFO):
            observer.on_run_started(RunId("rid"), Prompt("hi"))

        # Assert: the built observer logs JSON correlated by run_id under "agent".
        record = caplog.records[-1]
        assert record.name == LOGGER_NAME
        assert '"event": "run_started"' in record.getMessage()
        assert '"run_id": "rid"' in record.getMessage()
