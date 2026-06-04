from __future__ import annotations

import json
from pathlib import Path

from little_harness.domain.tool_result import ToolRunRequest
from little_harness.domain.values.text_values import ToolInput, ToolName
from little_harness_file_tools.write_file_tool import WriteFileTool


def write_request(payload: str) -> ToolRunRequest:
    return ToolRunRequest(ToolName("write_file"), ToolInput(payload))


class TestWriteFileTool:
    def test_advertises_a_sensitive_spec(self) -> None:
        # Act
        spec = WriteFileTool().spec

        # Assert
        assert spec.name == ToolName("write_file")
        assert spec.requires_approval is True

    def test_writes_content_and_creates_nested_parent_directories(
        self, tmp_path: Path
    ) -> None:
        # Arrange: two missing levels prove parents are created recursively.
        target = tmp_path / "deep" / "nested" / "out.txt"
        payload = json.dumps({"path": str(target), "content": "hello"})

        # Act
        result = WriteFileTool().run(write_request(payload))

        # Assert
        assert result.tool_name == ToolName("write_file")
        assert result.succeeded is True
        assert target.read_text(encoding="utf-8") == "hello"
        assert result.output.value == f"Wrote 5 characters to {target}."

    def test_overwrites_an_existing_file(self, tmp_path: Path) -> None:
        # Arrange: writing twice proves the existing directory is tolerated.
        target = tmp_path / "again" / "out.txt"
        WriteFileTool().run(
            write_request(json.dumps({"path": str(target), "content": "first"}))
        )

        # Act
        result = WriteFileTool().run(
            write_request(json.dumps({"path": str(target), "content": "second"}))
        )

        # Assert
        assert result.succeeded is True
        assert target.read_text(encoding="utf-8") == "second"

    def test_reports_a_failure_when_the_path_is_a_directory(
        self, tmp_path: Path
    ) -> None:
        # Act: writing onto an existing directory raises OSError, handled here.
        payload = json.dumps({"path": str(tmp_path), "content": "x"})
        result = WriteFileTool().run(write_request(payload))

        # Assert
        assert result.succeeded is False
        assert result.output.value.startswith("Write error:")

    def test_reports_a_failure_for_invalid_json(self) -> None:
        # Act
        result = WriteFileTool().run(write_request("not json"))

        # Assert
        assert result.tool_name == ToolName("write_file")
        assert result.succeeded is False
        assert result.output.value.startswith("Write error:")

    def test_reports_a_failure_for_a_missing_field(self, tmp_path: Path) -> None:
        # Arrange: content is required.
        payload = json.dumps({"path": str(tmp_path / "x.txt")})

        # Act
        result = WriteFileTool().run(write_request(payload))

        # Assert
        assert result.succeeded is False
        assert "Missing field 'content'" in result.output.value
