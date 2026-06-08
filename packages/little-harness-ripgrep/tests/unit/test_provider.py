from __future__ import annotations

from little_harness.domain.tool_result import ToolRunRequest
from little_harness.domain.values.text_values import ToolInput, ToolName
from little_harness_ripgrep.provider import build
from little_harness_ripgrep.ripgrep_tool import RipgrepTool


class TestBuild:
    def test_returns_a_ripgrep_tool(self) -> None:
        # Act
        tool = build()

        # Assert
        assert isinstance(tool, RipgrepTool)
        assert tool.spec.name == ToolName("ripgrep")
        assert tool.spec.requires_approval is False

    def test_built_tool_runs_through_a_real_search_backend(self) -> None:
        # A real PythonGrepSearch is wired in: running returns a result
        # (rather than crashing on a None search).
        request = ToolRunRequest(ToolName("ripgrep"), ToolInput("--version"))

        result = build().run(request)

        assert result.tool_name == ToolName("ripgrep")
