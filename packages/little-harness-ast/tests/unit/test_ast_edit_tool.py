from __future__ import annotations

import json
from pathlib import Path

from little_harness.domain.tool_result import ToolRunRequest
from little_harness.domain.values.text_values import ToolInput, ToolName
from little_harness_ast.ast_edit_tool import AstEditTool
from little_harness_ast.structural_match import StructuralMatch

from tests.unit.fakes import FakeSyntaxEngine

SOURCE = "print('hi')\nprint('bye')\n"
QUERY = "(call) @match"
# `print('hi')` occupies bytes 0..11 of SOURCE.
FIRST_CALL = StructuralMatch(1, 1, 0, 11, "print('hi')")


def edit_request(payload: str) -> ToolRunRequest:
    return ToolRunRequest(ToolName("ast_edit"), ToolInput(payload))


def payload_for(path: Path, replacement: str = "log()", query: str = QUERY) -> str:
    return json.dumps(
        {
            "path": str(path),
            "language": "python",
            "query": query,
            "replacement": replacement,
        }
    )


def write_source(tmp_path: Path) -> Path:
    target = tmp_path / "app.py"
    target.write_text(SOURCE, encoding="utf-8")
    return target


class TestAstEditTool:
    def test_advertises_a_sensitive_spec(self) -> None:
        spec = AstEditTool(FakeSyntaxEngine()).spec
        assert spec.name == ToolName("ast_edit")
        assert spec.requires_approval is True

    def test_replaces_the_unique_match_bytes(self, tmp_path: Path) -> None:
        # Arrange
        source = write_source(tmp_path)
        engine = FakeSyntaxEngine([FIRST_CALL])
        tool = AstEditTool(engine)

        # Act
        result = tool.run(edit_request(payload_for(source)))

        # Assert: only the matched node's bytes are replaced.
        assert result.tool_name == ToolName("ast_edit")
        assert result.succeeded is True
        assert source.read_text(encoding="utf-8") == "log()\nprint('bye')\n"
        assert result.output.value == f"Replaced 1 match (line 1) in {source}."
        assert engine.calls == [(SOURCE, "python", QUERY)]

    def test_rejects_when_there_is_no_match(self, tmp_path: Path) -> None:
        # Arrange
        source = write_source(tmp_path)
        tool = AstEditTool(FakeSyntaxEngine([]))

        # Act
        result = tool.run(edit_request(payload_for(source)))

        # Assert
        assert result.succeeded is False
        assert result.output.value == (
            f"ast-edit error: No match for query '{QUERY}' in {source}. "
            "Expected exactly one match."
        )

    def test_rejects_when_the_match_is_ambiguous(self, tmp_path: Path) -> None:
        # Arrange: two matches make a single structural edit ambiguous.
        source = write_source(tmp_path)
        engine = FakeSyntaxEngine([FIRST_CALL, StructuralMatch(2, 2, 12, 24, "x")])
        tool = AstEditTool(engine)

        # Act
        result = tool.run(edit_request(payload_for(source)))

        # Assert
        assert result.succeeded is False
        assert result.output.value == (
            f"ast-edit error: Query '{QUERY}' matched 2 nodes in {source}. "
            "Expected exactly one match."
        )

    def test_reports_a_missing_file(self, tmp_path: Path) -> None:
        # Arrange
        engine = FakeSyntaxEngine([FIRST_CALL])
        tool = AstEditTool(engine)

        # Act
        result = tool.run(edit_request(payload_for(tmp_path / "absent.py")))

        # Assert: it fails before reaching the engine.
        assert result.succeeded is False
        assert result.output.value.startswith("ast-edit error:")
        assert engine.calls == []

    def test_reports_invalid_json(self) -> None:
        result = AstEditTool(FakeSyntaxEngine()).run(edit_request("not json"))
        assert result.tool_name == ToolName("ast_edit")
        assert result.succeeded is False
        assert result.output.value.startswith("ast-edit error:")

    def test_reports_an_engine_error(self, tmp_path: Path) -> None:
        # Arrange
        source = write_source(tmp_path)
        engine = FakeSyntaxEngine(error=ValueError("Unsupported language: 'x'."))

        # Act
        result = AstEditTool(engine).run(edit_request(payload_for(source)))

        # Assert
        assert result.succeeded is False
        assert result.output.value == "ast-edit error: Unsupported language: 'x'."
