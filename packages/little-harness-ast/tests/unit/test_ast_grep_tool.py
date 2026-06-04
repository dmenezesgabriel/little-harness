from __future__ import annotations

import json
from pathlib import Path

from little_harness.domain.tool_result import ToolRunRequest
from little_harness.domain.values.text_values import ToolInput, ToolName
from little_harness_ast.ast_grep_tool import AstGrepTool
from little_harness_ast.structural_match import StructuralMatch

from tests.unit.fakes import FakeSyntaxEngine

QUERY = "(call) @match"


def grep_request(payload: str) -> ToolRunRequest:
    return ToolRunRequest(ToolName("ast_grep"), ToolInput(payload))


def payload_for(path: Path, query: str = QUERY) -> str:
    return json.dumps({"path": str(path), "language": "python", "query": query})


def write_source(tmp_path: Path, contents: str = "print('hi')\n") -> Path:
    target = tmp_path / "app.py"
    target.write_text(contents, encoding="utf-8")
    return target


class TestAstGrepTool:
    def test_advertises_a_safe_spec(self) -> None:
        spec = AstGrepTool(FakeSyntaxEngine()).spec
        assert spec.name == ToolName("ast_grep")
        assert spec.requires_approval is False

    def test_formats_matches_with_their_location(self, tmp_path: Path) -> None:
        # Arrange
        source = write_source(tmp_path)
        engine = FakeSyntaxEngine([StructuralMatch(2, 2, 0, 11, "print('hi')")])
        tool = AstGrepTool(engine)

        # Act
        result = tool.run(grep_request(payload_for(source)))

        # Assert: file contents, language, and query reach the engine.
        assert result.tool_name == ToolName("ast_grep")
        assert result.succeeded is True
        assert result.output.value == f"{source}:line 2: print('hi')"
        assert engine.calls == [("print('hi')\n", "python", QUERY)]

    def test_joins_multiple_matches(self, tmp_path: Path) -> None:
        # Arrange
        source = write_source(tmp_path)
        engine = FakeSyntaxEngine(
            [StructuralMatch(1, 1, 0, 1, "a"), StructuralMatch(3, 4, 5, 9, "b")]
        )

        # Act
        result = AstGrepTool(engine).run(grep_request(payload_for(source)))

        # Assert
        assert result.output.value == f"{source}:line 1: a\n{source}:lines 3-4: b"

    def test_reports_no_matches(self, tmp_path: Path) -> None:
        # Arrange
        source = write_source(tmp_path)

        # Act
        result = AstGrepTool(FakeSyntaxEngine([])).run(
            grep_request(payload_for(source))
        )

        # Assert
        assert result.tool_name == ToolName("ast_grep")
        assert result.succeeded is True
        assert result.output.value == "No matches found."

    def test_reports_a_missing_file(self, tmp_path: Path) -> None:
        # Arrange
        engine = FakeSyntaxEngine([StructuralMatch(1, 1, 0, 1, "x")])

        # Act
        result = AstGrepTool(engine).run(grep_request(payload_for(tmp_path / "no.py")))

        # Assert: it fails before reaching the engine.
        assert result.succeeded is False
        assert result.output.value.startswith("ast-grep error:")
        assert engine.calls == []

    def test_reports_invalid_json(self) -> None:
        result = AstGrepTool(FakeSyntaxEngine()).run(grep_request("not json"))
        assert result.tool_name == ToolName("ast_grep")
        assert result.succeeded is False
        assert result.output.value.startswith("ast-grep error:")

    def test_reports_an_engine_error(self, tmp_path: Path) -> None:
        # Arrange: the engine rejects a bad query with a ValueError.
        source = write_source(tmp_path)
        engine = FakeSyntaxEngine(error=ValueError("Invalid query: '('."))

        # Act
        result = AstGrepTool(engine).run(grep_request(payload_for(source, query="(")))

        # Assert
        assert result.succeeded is False
        assert result.output.value == "ast-grep error: Invalid query: '('."
