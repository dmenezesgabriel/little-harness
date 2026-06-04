from __future__ import annotations

from pathlib import Path

from little_harness.domain.tool_result import ToolRunRequest
from little_harness.domain.values.text_values import ToolInput, ToolName
from little_harness_file_tools.read_file_tool import ReadFileTool


def read_request(path: str) -> ToolRunRequest:
    return ToolRunRequest(ToolName("read_file"), ToolInput(path))


class TestReadFileTool:
    def test_advertises_a_safe_spec(self) -> None:
        # Act
        spec = ReadFileTool().spec

        # Assert: reading is safe, so it never asks for approval.
        assert spec.name == ToolName("read_file")
        assert spec.requires_approval is False

    def test_returns_the_file_contents(self, tmp_path: Path) -> None:
        # Arrange
        target = tmp_path / "note.txt"
        target.write_text("line one\nline two\n", encoding="utf-8")

        # Act
        result = ReadFileTool().run(read_request(str(target)))

        # Assert
        assert result.tool_name == ToolName("read_file")
        assert result.succeeded is True
        assert result.output.value == "line one\nline two\n"

    def test_reports_a_failure_when_the_file_is_missing(self, tmp_path: Path) -> None:
        # Act: a missing file raises OSError, which must be reported, not raised.
        result = ReadFileTool().run(read_request(str(tmp_path / "absent.txt")))

        # Assert
        assert result.tool_name == ToolName("read_file")
        assert result.succeeded is False
        assert result.output.value.startswith("Read error:")

    def test_reports_a_failure_for_a_path_with_a_null_byte(self) -> None:
        # Act: a NUL byte in the path raises ValueError, also handled gracefully.
        result = ReadFileTool().run(read_request("bad\x00path.txt"))

        # Assert
        assert result.succeeded is False
        assert result.output.value.startswith("Read error:")
