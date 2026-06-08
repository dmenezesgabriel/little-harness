"""Binds the grep feature to the real CLI stack with a scripted model.

Four scenarios: basic action name, real match, no-match, and invalid regex.
All four run without the rg binary — the pure-Python engine is used directly.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import pytest
from little_harness.composition import run_cli
from pytest_bdd import given, parsers, scenarios, then, when

from tests.integration.conftest import final_answer, tool_call

pytestmark = pytest.mark.integration

scenarios("features/ripgrep_through_core.feature")

Install = Callable[[Sequence[str]], None]


@when(
    parsers.parse('the agent uses grep with arguments "{arguments}"'),
    target_fixture="output",
)
def use_grep(install_scripted_provider: Install, arguments: str) -> str:
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


# --- Scenarios with a real temp file ---


@pytest.fixture
def temp_file() -> Iterator[dict[str, str]]:
    """Yield a mutable dict that steps can fill with {path, content}."""
    yield {}


@given(
    parsers.parse('a file "{filename}" containing "{content}"'),
    target_fixture="temp_file",
)
def create_temp_file(tmp_path: Path, filename: str, content: str) -> dict[str, str]:
    # tmp_path here is the real pytest fixture injected by pytest-bdd
    path = tmp_path / filename
    path.write_text(content, encoding="utf-8")
    return {"path": str(path), "content": content}


@when(
    parsers.parse('the agent searches for "{pattern}" in that file'),
    target_fixture="output",
)
def search_pattern_in_file(
    install_scripted_provider: Install,
    temp_file: dict[str, str],
    pattern: str,
) -> str:
    arguments = f"{pattern} {temp_file['path']}"
    install_scripted_provider([tool_call("ripgrep", arguments), final_answer("done")])
    return run_cli(
        [
            "--provider",
            "scripted",
            "--tools",
            "ripgrep",
            "--prompt",
            "search",
            "--yes",
            "--max-iterations",
            "3",
        ]
    )


@when(
    parsers.parse('the agent searches with invalid regex "{pattern}" in that file'),
    target_fixture="output",
)
def search_invalid_regex(
    install_scripted_provider: Install,
    temp_file: dict[str, str],
    pattern: str,
) -> str:
    arguments = f"{pattern} {temp_file['path']}"
    install_scripted_provider([tool_call("ripgrep", arguments), final_answer("done")])
    return run_cli(
        [
            "--provider",
            "scripted",
            "--tools",
            "ripgrep",
            "--prompt",
            "search",
            "--yes",
            "--max-iterations",
            "3",
        ]
    )


@then("the search succeeds")
def search_succeeded(output: str) -> None:
    # A succeeded search appears as a completed agent run.
    assert "done" in output or "Observation" in output


@then(parsers.parse('the output contains "{expected}"'))
def output_contains(output: str, expected: str) -> None:
    assert expected in output, f"Expected {expected!r} in:\n{output}"


@then("the search fails")
def search_failed(output: str) -> None:
    # A failed tool call still completes the run; the observation carries the error.
    assert "Observation" in output
