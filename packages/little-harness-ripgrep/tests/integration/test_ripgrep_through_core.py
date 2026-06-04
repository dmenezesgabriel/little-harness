"""Binds the ripgrep feature to the real CLI stack with a scripted model."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pytest
from little_harness.composition import run_cli
from pytest_bdd import parsers, scenarios, then, when

from tests.integration.conftest import final_answer, tool_call

pytestmark = pytest.mark.integration

scenarios("features/ripgrep_through_core.feature")

Install = Callable[[Sequence[str]], None]


@when(
    parsers.parse('the agent uses ripgrep with arguments "{arguments}"'),
    target_fixture="output",
)
def use_ripgrep(install_scripted_provider: Install, arguments: str) -> str:
    install_scripted_provider([tool_call("ripgrep", arguments), final_answer("done")])
    return run_cli(
        [
            "--provider",
            "scripted",
            "--tools",
            "ripgrep",
            "--prompt",
            "search the code",
            "--yes",
            "--max-iterations",
            "3",
        ]
    )


@then(parsers.parse('the run output shows action "{action}"'))
def output_shows_action(output: str, action: str) -> None:
    assert f"Action: {action}" in output
