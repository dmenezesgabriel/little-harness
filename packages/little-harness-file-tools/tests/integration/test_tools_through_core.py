"""Binds the filesystem-tools feature to the real CLI stack with a scripted model."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
from little_harness.composition import run_cli
from pytest_bdd import given, parsers, scenarios, then, when

from tests.integration.conftest import final_answer, tool_call

pytestmark = pytest.mark.integration

scenarios("features/tools_through_core.feature")

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
            "do the task",
            "--yes",
            "--max-iterations",
            "3",
        ]
    )


@given(parsers.parse('a file "{name}" containing "{content}"'))
def existing_file(tmp_path: Path, name: str, content: str) -> None:
    (tmp_path / name).write_text(content, encoding="utf-8")


@when(parsers.parse('the agent uses read_file on "{name}"'), target_fixture="output")
def use_read_file(install_scripted_provider: Install, tmp_path: Path, name: str) -> str:
    return run_tool(install_scripted_provider, "read_file", str(tmp_path / name))


@when(
    parsers.parse('the agent uses write_file to write "{content}" to "{name}"'),
    target_fixture="output",
)
def use_write_file(
    install_scripted_provider: Install, tmp_path: Path, content: str, name: str
) -> str:
    payload = json.dumps({"path": str(tmp_path / name), "content": content})
    return run_tool(install_scripted_provider, "write_file", payload)


@when(
    parsers.parse(
        'the agent uses edit_file to replace "{old}" with "{new}" in "{name}"'
    ),
    target_fixture="output",
)
def use_edit_file(
    install_scripted_provider: Install,
    tmp_path: Path,
    old: str,
    new: str,
    name: str,
) -> str:
    payload = json.dumps({"path": str(tmp_path / name), "old": old, "new": new})
    return run_tool(install_scripted_provider, "edit_file", payload)


@when(parsers.parse('the agent uses bash to run "{command}"'), target_fixture="output")
def use_bash(install_scripted_provider: Install, command: str) -> str:
    return run_tool(install_scripted_provider, "bash", command)


@then(parsers.parse('the run output contains "{text}"'))
def output_contains(output: str, text: str) -> None:
    assert text in output


@then(parsers.parse('the run output shows action "{action}"'))
def output_shows_action(output: str, action: str) -> None:
    assert f"Action: {action}" in output


@then(parsers.parse('the file "{name}" contains "{content}"'))
def file_contains(tmp_path: Path, name: str, content: str) -> None:
    assert (tmp_path / name).read_text(encoding="utf-8") == content
