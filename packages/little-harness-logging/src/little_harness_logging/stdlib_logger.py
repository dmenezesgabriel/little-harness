"""StructuredLogger adapter over the standard library `logging` module."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping


class StdlibStructuredLogger:
    """Writes one JSON record per event through an injected stdlib logger.

    Example:
        StdlibStructuredLogger(logging.getLogger("agent")).log("started", {})

    """

    def __init__(self, logger: logging.Logger) -> None:
        """See class docstring for argument descriptions."""
        self._logger = logger

    def log(self, event: str, fields: Mapping[str, object]) -> None:
        """Emit a JSON log record for a named event with structured fields."""
        self._logger.info(json.dumps({"event": event, **fields}, default=str))


def create_structured_logger(name: str) -> StdlibStructuredLogger:
    """Create a StdlibStructuredLogger configured to emit JSON to stderr."""
    logger = logging.getLogger(name)
    configure_stderr_emission(logger)
    return StdlibStructuredLogger(logger)


def configure_stderr_emission(logger: logging.Logger) -> None:
    """Emit records to stderr at INFO without requiring `logging.basicConfig`.

    stderr keeps structured logs off stdout, where the plain-text CLI answer goes.
    Idempotent: a second call does not add a duplicate handler.
    """
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return

    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
