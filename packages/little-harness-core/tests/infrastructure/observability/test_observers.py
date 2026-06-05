from __future__ import annotations

from little_harness.application.ports.agent_observer import AgentObserver
from little_harness.domain.values.numeric_values import Iteration
from little_harness.domain.values.text_values import Prompt, RunId
from little_harness.infrastructure.observability.null_observer import NullObserver

RUN_ID = RunId("run-1")


class TestNullObserver:
    def test_conforms_to_the_observer_port_and_does_nothing(self) -> None:
        # Arrange: the explicit annotation forces a protocol-conformance check.
        observer: AgentObserver = NullObserver()

        # Act / Assert: every call is a no-op that returns None.
        assert observer.on_run_started(RUN_ID, Prompt("q")) is None
        assert observer.on_repair(RUN_ID, Iteration(1), ValueError("x")) is None
