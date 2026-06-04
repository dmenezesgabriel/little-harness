from __future__ import annotations

import json
import logging

import pytest
from little_harness.infrastructure.observability.stdlib_logger import (
    StdlibStructuredLogger,
    configure_stderr_emission,
    create_structured_logger,
)


class TestStdlibStructuredLogger:
    def test_emits_one_json_record_per_event(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Arrange
        logger = create_structured_logger("little_harness_test_logger")

        # Act
        with caplog.at_level(logging.INFO, logger="little_harness_test_logger"):
            logger.log("tool_invoked", {"tool": "calculator", "ok": True})

        # Assert
        assert json.loads(caplog.records[0].getMessage()) == {
            "event": "tool_invoked",
            "tool": "calculator",
            "ok": True,
        }

    def test_created_logger_emits_to_stderr_without_external_configuration(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # Arrange: no logging.basicConfig — the factory must configure emission.
        logger = create_structured_logger("little_harness_stderr_logger")

        # Act
        logger.log("run_started", {"run_id": "abc"})

        # Assert
        assert '"event": "run_started"' in capsys.readouterr().err

    def test_configure_adds_one_info_stream_handler_and_is_idempotent(self) -> None:
        # Arrange: a fresh logger with no handlers exercises the full branch.
        logger = logging.getLogger("little_harness_configure_test")
        logger.handlers.clear()

        # Act
        configure_stderr_emission(logger)
        configure_stderr_emission(logger)  # second call must not duplicate

        # Assert
        assert logger.level == logging.INFO
        assert len(logger.handlers) == 1
        handler = logger.handlers[0]
        assert isinstance(handler, logging.StreamHandler)
        assert handler.level == logging.INFO

    def test_serializes_non_json_values_via_str(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Arrange
        logger = StdlibStructuredLogger(
            logging.getLogger("little_harness_test_logger2")
        )

        # Act: a Path is not natively JSON-serializable; default=str handles it.
        with caplog.at_level(logging.INFO, logger="little_harness_test_logger2"):
            logger.log("started", {"value": ValueError("boom")})

        # Assert
        assert json.loads(caplog.records[0].getMessage()) == {
            "event": "started",
            "value": "boom",
        }
