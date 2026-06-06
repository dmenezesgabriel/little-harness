"""Runs the JSON policy through the real core repair loop."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence

import pytest
from little_harness.composition import run_cli

pytestmark = pytest.mark.integration

Install = Callable[[Sequence[str]], None]


def test_policy_repairs_invalid_model_output_through_the_core(
    install_scripted_provider: Install,
) -> None:
    # Arrange
    install_scripted_provider(
        ["not json", json.dumps({"action": "final", "answer": "recovered"})]
    )

    # Act
    output = run_cli(
        [
            "--provider",
            "scripted",
            "--policy",
            "json",
            "--prompt",
            "answer plainly",
            "--max-iterations",
            "2",
        ]
    )

    # Assert
    assert "recovered" in output
    assert "Action: repair" in output
