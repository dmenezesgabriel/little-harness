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
        self._logger = logger

    def log(self, event: str, fields: Mapping[str, object]) -> None:
        self._logger.info(json.dumps({"event": event, **fields}, default=str))


def create_structured_logger(name: str) -> StdlibStructuredLogger:
    return StdlibStructuredLogger(logging.getLogger(name))
