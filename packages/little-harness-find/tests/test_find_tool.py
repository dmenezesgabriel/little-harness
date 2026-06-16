"""Tests for FindTool — pure-Python glob search."""

from __future__ import annotations

from pathlib import Path

import pytest
from little_harness.domain.tool_result import ToolRunRequest, ToolRunResult
from little_harness.domain.values.text_values import ToolInput, ToolName

from little_harness_find.find_tool import FindTool


def find_request(raw: str) -> ToolRunRequest:
    return ToolRunRequest(ToolName("find"), ToolInput(raw))


def make_tree(tmp_path: Path, *paths: str) -> Path:
    for p in paths:
        file = tmp_path / p
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("content\n")
    return tmp_path


class TestFindToolSpec:
    def test_advertises_the_find_name(self) -> None:
        spec = FindTool().spec
        assert spec.name == ToolName("find")

    def test_description_is_not_empty(self) -> None:
        spec = FindTool().spec
        assert spec.description

    def test_input_schema_has_description(self) -> None:
        spec = FindTool().spec
        assert spec.input_schema.description

    def test_has_examples(self) -> None:
        spec = FindTool().spec
        assert not spec.input_schema.examples.is_empty()

    def test_requires_no_approval(self) -> None:
        spec = FindTool().spec
        assert spec.requires_approval is False


class TestFindToolRun:
    def test_finds_matching_files(self, tmp_path: Path) -> None:
        make_tree(tmp_path, "a.txt", "b.py", "c.txt")
        tool = FindTool()
        request = find_request(
            f'{{"pattern": "*.txt", "path": "{tmp_path!s}"}}'
        )
        result = tool.run(request)
        assert result.succeeded is True
        lines = result.output.value.splitlines()
        assert "a.txt" in lines
        assert "b.py" not in lines
        assert "c.txt" in lines

    def test_finds_nested_files(self, tmp_path: Path) -> None:
        make_tree(tmp_path, "src/a.txt", "src/b.py", "tests/c.txt")
        tool = FindTool()
        request = find_request(
            f'{{"pattern": "**/*.txt", "path": "{tmp_path!s}"}}'
        )
        result = tool.run(request)
        assert result.succeeded is True
        lines = result.output.value.splitlines()
        assert "src/a.txt" in lines
        assert "tests/c.txt" in lines

    def test_returns_empty_when_no_match(self, tmp_path: Path) -> None:
        make_tree(tmp_path, "a.py")
        tool = FindTool()
        request = find_request(
            f'{{"pattern": "*.md", "path": "{tmp_path!s}"}}'
        )
        result = tool.run(request)
        assert result.succeeded is True
        assert result.output.value.strip() == ""

    def test_defaults_to_current_directory(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "hello.txt").write_text("content\n")
        tool = FindTool()
        request = find_request('{"pattern": "*.txt"}')
        result = tool.run(request)
        assert result.succeeded is True
        assert "hello.txt" in result.output.value

    def test_applies_result_limit(self, tmp_path: Path) -> None:
        for i in range(20):
            (tmp_path / f"file_{i:02d}.txt").write_text("content\n")
        tool = FindTool()
        request = find_request(
            f'{{"pattern": "*.txt", "path": "{tmp_path!s}", "limit": 5}}'
        )
        result = tool.run(request)
        assert result.succeeded is True
        lines = [ln for ln in result.output.value.splitlines() if ln]
        assert len(lines) == 5

    def test_skips_git_directory_by_default(self, tmp_path: Path) -> None:
        make_tree(tmp_path, ".git/HEAD", "src/a.py")
        tool = FindTool()
        request = find_request(
            f'{{"pattern": "**/*", "path": "{tmp_path!s}"}}'
        )
        result = tool.run(request)
        lines = result.output.value.splitlines()
        assert "src/a.py" in lines
        assert ".git/HEAD" not in lines
        assert ".git" not in lines

    def test_skips_node_modules_by_default(self, tmp_path: Path) -> None:
        make_tree(tmp_path, "node_modules/lodash/index.js", "src/app.js")
        tool = FindTool()
        request = find_request(
            f'{{"pattern": "**/*.js", "path": "{tmp_path!s}"}}'
        )
        result = tool.run(request)
        lines = result.output.value.splitlines()
        assert "src/app.js" in lines
        assert "node_modules/lodash/index.js" not in lines

    def test_returns_error_for_invalid_json(self) -> None:
        tool = FindTool()
        request = find_request("not-json")
        result = tool.run(request)
        assert result.succeeded is False
        assert "Find error" in result.output.value

    def test_returns_error_for_non_existent_path(self, tmp_path: Path) -> None:
        tool = FindTool()
        missing = tmp_path / "does-not-exist"
        request = find_request(
            f'{{"pattern": "*.txt", "path": "{missing!s}"}}'
        )
        result = tool.run(request)
        assert result.succeeded is False
        assert "Find error" in result.output.value
        assert "does-not-exist" in result.output.value


def test_build_returns_find_tool() -> None:
    from little_harness_find.provider import build
    from little_harness_find.find_tool import FindTool
    tool = build()
    assert isinstance(tool, FindTool)
    assert tool.spec.name == ToolName("find")
