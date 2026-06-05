"""Entry-point builder for the structured-logging observer.

Registered under the `little_harness.observers` group as `logging`. The core
composition root calls `build()` when this observer is selected (`--observer
logging`, or its `--log` shorthand) and threads the result through the agent
loop, so observability is added without any core edit.

Example:
    observer = build()
"""

from __future__ import annotations

from little_harness.application.ports.agent_observer import AgentObserver

from little_harness_logging.stdlib_logger import create_structured_logger
from little_harness_logging.structured_logging_observer import StructuredLoggingObserver

# Every record is emitted under this logger name so a run's events correlate.
LOGGER_NAME = "agent"


def build() -> AgentObserver:
    return StructuredLoggingObserver(create_structured_logger(LOGGER_NAME))
