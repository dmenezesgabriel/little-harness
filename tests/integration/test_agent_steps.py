"""Step definitions binding the agent_calculator feature to the real stack."""

from __future__ import annotations

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from local_llm.composition import run_cli

pytestmark = pytest.mark.integration

scenarios("features/agent_calculator.feature")


@given("a local agent")
def local_agent() -> None:
    return None


@when(parsers.parse('I ask "{question}"'), target_fixture="answer")
def ask_the_agent(question: str) -> str:
    return run_cli(
        ["--prompt", question, "--max-tokens", "256", "--max-iterations", "3"]
    )


@then(parsers.parse('the answer contains "{expected}"'))
def answer_contains(answer: str, expected: str) -> None:
    assert expected in answer
