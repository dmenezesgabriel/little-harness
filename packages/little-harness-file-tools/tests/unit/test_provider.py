from __future__ import annotations

from little_harness.domain.tool_result import ToolRunRequest
from little_harness.domain.values.text_values import ToolInput, ToolName
from little_harness_file_tools.bash_tool import BashTool
from little_harness_file_tools.edit_file_tool import EditFileTool
from little_harness_file_tools.provider import (
    build_bash,
    build_edit_file,
    build_read_file,
    build_write_file,
)
from little_harness_file_tools.read_file_tool import ReadFileTool
from little_harness_file_tools.write_file_tool import WriteFileTool


class TestBuilders:
    def test_builds_a_read_file_tool(self) -> None:
        tool = build_read_file()
        assert isinstance(tool, ReadFileTool)
        assert tool.spec.name == ToolName("read_file")

    def test_builds_a_write_file_tool(self) -> None:
        tool = build_write_file()
        assert isinstance(tool, WriteFileTool)
        assert tool.spec.name == ToolName("write_file")

    def test_builds_an_edit_file_tool(self) -> None:
        tool = build_edit_file()
        assert isinstance(tool, EditFileTool)
        assert tool.spec.name == ToolName("edit_file")

    def test_builds_a_bash_tool_that_requires_approval(self) -> None:
        tool = build_bash()
        assert isinstance(tool, BashTool)
        assert tool.spec.name == ToolName("bash")
        assert tool.spec.requires_approval is True

    def test_built_bash_tool_runs_with_a_real_runner_and_guardrail(self) -> None:
        # A real command must succeed (real runner) and the guardrail must be
        # wired (so it inspects the command rather than crashing on a missing one).
        request = ToolRunRequest(ToolName("bash"), ToolInput("echo provider-ok"))

        result = build_bash().run(request)

        assert result.succeeded is True
        assert "provider-ok" in result.output.value
