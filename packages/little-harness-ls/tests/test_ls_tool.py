"""Tests for LsTool — pure-Python directory listing."""

from __future__ import annotations

import stat as stat_module
from pathlib import Path

import pytest
from little_harness.domain.tool_result import ToolRunRequest, ToolRunResult
from little_harness.domain.values.text_values import ToolInput, ToolName

from little_harness_ls.ls_tool import LsTool


def ls_request(raw: str) -> ToolRunRequest:
    return ToolRunRequest(ToolName("ls"), ToolInput(raw))


def make_tree(tmp_path: Path, *names: str) -> Path:
    for name in names:
        entry = tmp_path / name
        if name.endswith("/"):
            entry.mkdir(parents=True, exist_ok=True)
        else:
            entry.parent.mkdir(parents=True, exist_ok=True)
            entry.write_text("content\n")
    return tmp_path


class TestLsToolSpec:
    def test_advertises_the_ls_name(self) -> None:
        spec = LsTool().spec
        assert spec.name == ToolName("ls")

    def test_description_is_not_empty(self) -> None:
        spec = LsTool().spec
        assert spec.description

    def test_input_schema_has_description(self) -> None:
        spec = LsTool().spec
        assert spec.input_schema.description

    def test_has_examples(self) -> None:
        spec = LsTool().spec
        assert not spec.input_schema.examples.is_empty()

    def test_requires_no_approval(self) -> None:
        spec = LsTool().spec
        assert spec.requires_approval is False


class TestLsToolRun:
    def test_lists_directory_contents(self, tmp_path: Path) -> None:
        make_tree(tmp_path, "a.txt", "b.py")
        tool = LsTool()
        request = ls_request(f'{{"path": "{tmp_path!s}"}}')
        result = tool.run(request)
        assert result.succeeded is True
        lines = result.output.value.splitlines()
        assert "a.txt" in lines
        assert "b.py" in lines

    def test_marks_directories_with_trailing_slash(self, tmp_path: Path) -> None:
        make_tree(tmp_path, "file.txt", "subdir/")
        tool = LsTool()
        request = ls_request(f'{{"path": "{tmp_path!s}"}}')
        result = tool.run(request)
        lines = result.output.value.splitlines()
        assert "file.txt" in lines
        assert "subdir/" in lines

    def test_includes_dotfiles(self, tmp_path: Path) -> None:
        make_tree(tmp_path, ".hidden", "visible")
        tool = LsTool()
        request = ls_request(f'{{"path": "{tmp_path!s}"}}')
        result = tool.run(request)
        lines = result.output.value.splitlines()
        assert ".hidden" in lines
        assert "visible" in lines

    def test_sorts_entries_case_insensitive(self, tmp_path: Path) -> None:
        make_tree(tmp_path, "B.txt", "a.txt", "c.txt")
        tool = LsTool()
        request = ls_request(f'{{"path": "{tmp_path!s}"}}')
        result = tool.run(request)
        lines = result.output.value.splitlines()
        assert lines == ["a.txt", "B.txt", "c.txt"]

    def test_applies_entry_limit(self, tmp_path: Path) -> None:
        for i in range(20):
            (tmp_path / f"file_{i:02d}.txt").write_text("content\n")
        tool = LsTool()
        request = ls_request(
            f'{{"path": "{tmp_path!s}", "limit": 5}}'
        )
        result = tool.run(request)
        assert result.succeeded is True
        lines = [ln for ln in result.output.value.splitlines() if ln]
        assert len(lines) == 5

    def test_defaults_to_current_directory(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "hello.txt").write_text("content\n")
        tool = LsTool()
        request = ls_request("{}")
        result = tool.run(request)
        assert result.succeeded is True
        assert "hello.txt" in result.output.value

    def test_returns_error_for_non_existent_path(self, tmp_path: Path) -> None:
        tool = LsTool()
        missing = tmp_path / "does-not-exist"
        request = ls_request(f'{{"path": "{missing!s}"}}')
        result = tool.run(request)
        assert result.succeeded is False
        assert "Ls error" in result.output.value

    def test_returns_error_for_path_to_file(self, tmp_path: Path) -> None:
        make_tree(tmp_path, "a_file.txt")
        tool = LsTool()
        file_path = tmp_path / "a_file.txt"
        request = ls_request(f'{{"path": "{file_path!s}"}}')
        result = tool.run(request)
        assert result.succeeded is False
        assert "Ls error" in result.output.value

    def test_returns_error_for_invalid_json(self) -> None:
        tool = LsTool()
        request = ls_request("not-json")
        result = tool.run(request)
        assert result.succeeded is False
        assert "Ls error" in result.output.value

    def test_returns_error_when_limit_not_int(self, tmp_path: Path) -> None:
        tool = LsTool()
        request = ls_request(f'{{"path": "{tmp_path!s}", "limit": "not-a-number"}}')
        result = tool.run(request)
        assert result.succeeded is False
        assert "Ls error" in result.output.value

    def test_skips_entries_that_fail_to_stat(self, tmp_path: Path) -> None:
        make_tree(tmp_path, "good.txt")
        broken = tmp_path / "broken_link"
        broken.symlink_to("/nonexistent")
        tool = LsTool()
        request = ls_request(f'{{"path": "{tmp_path!s}"}}')
        result = tool.run(request)
        assert result.succeeded is True
        lines = result.output.value.splitlines()
        assert "good.txt" in lines


def test_build_returns_ls_tool() -> None:
    from little_harness_ls.provider import build
    from little_harness_ls.ls_tool import LsTool
    tool = build()
    assert isinstance(tool, LsTool)
    assert tool.spec.name == ToolName("ls")
