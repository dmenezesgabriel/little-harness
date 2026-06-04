from __future__ import annotations

import json
from pathlib import Path

from little_harness.domain.tool_result import ToolRunRequest
from little_harness.domain.values.text_values import ToolInput, ToolName
from little_harness_ast.ast_edit_tool import AstEditTool
from little_harness_ast.ast_grep_tool import AstGrepTool
from little_harness_ast.provider import build_ast_edit, build_ast_grep


class TestBuilders:
    def test_builds_an_ast_grep_tool(self) -> None:
        tool = build_ast_grep()
        assert isinstance(tool, AstGrepTool)
        assert tool.spec.name == ToolName("ast_grep")
        assert tool.spec.requires_approval is False

    def test_builds_an_ast_edit_tool(self) -> None:
        tool = build_ast_edit()
        assert isinstance(tool, AstEditTool)
        assert tool.spec.name == ToolName("ast_edit")
        assert tool.spec.requires_approval is True

    def test_built_grep_tool_searches_with_the_real_engine(
        self, tmp_path: Path
    ) -> None:
        # A real TreeSitterEngine is wired in (None would crash on run).
        source = tmp_path / "app.py"
        source.write_text("print('wired')\n", encoding="utf-8")
        payload = json.dumps(
            {"path": str(source), "language": "python", "query": "(call) @match"}
        )

        result = build_ast_grep().run(
            ToolRunRequest(ToolName("ast_grep"), ToolInput(payload))
        )

        assert result.succeeded is True
        assert "print('wired')" in result.output.value

    def test_built_edit_tool_rewrites_with_the_real_engine(
        self, tmp_path: Path
    ) -> None:
        # Arrange
        source = tmp_path / "app.py"
        source.write_text("print('old')\n", encoding="utf-8")
        payload = json.dumps(
            {
                "path": str(source),
                "language": "python",
                "query": "(call) @match",
                "replacement": "log('new')",
            }
        )

        # Act
        result = build_ast_edit().run(
            ToolRunRequest(ToolName("ast_edit"), ToolInput(payload))
        )

        # Assert
        assert result.succeeded is True
        assert source.read_text(encoding="utf-8") == "log('new')\n"
