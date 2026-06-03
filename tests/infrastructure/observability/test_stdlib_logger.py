from __future__ import annotations

import json
import logging

import pytest

from local_llm.infrastructure.observability.stdlib_logger import (
    StdlibStructuredLogger,
    create_structured_logger,
)


class TestStdlibStructuredLogger:
    def test_emits_one_json_record_per_event(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Arrange
        logger = create_structured_logger("local_llm_test_logger")

        # Act
        with caplog.at_level(logging.INFO, logger="local_llm_test_logger"):
            logger.log("tool_invoked", {"tool": "calculator", "ok": True})

        # Assert
        assert json.loads(caplog.records[0].getMessage()) == {
            "event": "tool_invoked",
            "tool": "calculator",
            "ok": True,
        }

    def test_serializes_non_json_values_via_str(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Arrange
        logger = StdlibStructuredLogger(logging.getLogger("local_llm_test_logger2"))

        # Act: a Path is not natively JSON-serializable; default=str handles it.
        with caplog.at_level(logging.INFO, logger="local_llm_test_logger2"):
            logger.log("started", {"value": ValueError("boom")})

        # Assert
        assert json.loads(caplog.records[0].getMessage()) == {
            "event": "started",
            "value": "boom",
        }
