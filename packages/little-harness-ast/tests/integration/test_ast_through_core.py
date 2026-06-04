"""Binds the AST-tools feature to the real CLI stack with a scripted model."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
from little_harness.composition import run_cli
from pytest_bdd import given, parsers, scenarios, then, when

from tests.integration.conftest import final_answer, tool_call

pytestmark = pytest.mark.integration

scenarios("features/ast_through_core.feature")

Install = Callable[[Sequence[str]], None]


def run_tool(install: Install, tool_name: str, tool_input: str) -> str:
    install([tool_call(tool_name, tool_input), final_answer("done")])
    return run_cli(
        [
            "--provider",
            "scripted",
            "--tools",
            tool_name,
            "--prompt",
            "work on the code",
            "--yes",
            "--max-iterations",
            "3",
        ]
    )


@given(parsers.parse('a python file "{name}" containing "{content}"'))
def python_file(tmp_path: Path, name: str, content: str) -> None:
    (tmp_path / name).write_text(content + "\n", encoding="utf-8")


@when(
    parsers.parse('the agent uses ast_grep on "{name}" with query "{query}"'),
    target_fixture="output",
)
def use_ast_grep(
    install_scripted_provider: Install, tmp_path: Path, name: str, query: str
) -> str:
    payload = json.dumps(
        {"path": str(tmp_path / name), "language": "python", "query": query}
    )
    return run_tool(install_scripted_provider, "ast_grep", payload)


@when(
    parsers.parse(
        'the agent uses ast_edit on "{name}" with query "{query}" '
        'and replacement "{replacement}"'
    ),
    target_fixture="output",
)
def use_ast_edit(
    install_scripted_provider: Install,
    tmp_path: Path,
    name: str,
    query: str,
    replacement: str,
) -> str:
    payload = json.dumps(
        {
            "path": str(tmp_path / name),
            "language": "python",
            "query": query,
            "replacement": replacement,
        }
    )
    return run_tool(install_scripted_provider, "ast_edit", payload)


@then(parsers.parse('the run output contains "{text}"'))
def output_contains(output: str, text: str) -> None:
    assert text in output


@then(parsers.parse('the run output shows action "{action}"'))
def output_shows_action(output: str, action: str) -> None:
    assert f"Action: {action}" in output


@then(parsers.parse('the file "{name}" contains "{content}"'))
def file_contains(tmp_path: Path, name: str, content: str) -> None:
    assert content in (tmp_path / name).read_text(encoding="utf-8")
