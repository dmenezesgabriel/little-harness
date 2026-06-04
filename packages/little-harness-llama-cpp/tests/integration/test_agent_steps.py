"""Step definitions binding the agent_calculator feature to the real stack."""

from __future__ import annotations

from pathlib import Path

import pytest
from little_harness.composition import run_cli
from pytest_bdd import given, parsers, scenarios, then, when

pytestmark = pytest.mark.integration

scenarios("features/agent_calculator.feature")


@given("a local agent")
def local_agent() -> None:
    return None


@when(parsers.parse('I ask "{question}"'), target_fixture="answer")
def ask_the_agent(question: str, model_path: Path) -> str:
    return run_cli(
        [
            "--provider",
            "llama_cpp",
            "--prompt",
            question,
            "-o",
            f"model_path={model_path}",
            "--max-tokens",
            "256",
            "--max-iterations",
            "3",
        ]
    )


@then(parsers.parse('the answer contains "{expected}"'))
def answer_contains(answer: str, expected: str) -> None:
    assert expected in answer
