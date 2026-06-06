"""Runs the calculator tool through the real core with a scripted model."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence

import pytest
from little_harness.composition import run_cli

pytestmark = pytest.mark.integration

Install = Callable[[Sequence[str]], None]


def test_calculator_runs_through_the_agent_core(
    install_scripted_provider: Install,
) -> None:
    # Arrange
    install_scripted_provider(
        [
            json.dumps({"action": "calculator", "input": "144 / 12"}),
            json.dumps({"action": "final", "answer": "12"}),
        ]
    )

    # Act
    output = run_cli(
        [
            "--provider",
            "scripted",
            "--tools",
            "calculator",
            "--prompt",
            "calculate it",
            "--max-iterations",
            "3",
        ]
    )

    # Assert
    assert "12" in output
    assert "Action: calculator" in output
