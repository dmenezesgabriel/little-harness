from __future__ import annotations

import json
from pathlib import Path

from little_harness.domain.tool_result import ToolRunRequest
from little_harness.domain.values.text_values import ToolInput, ToolName
from little_harness_file_tools.edit_file_tool import EditFileTool


def edit_request(payload: str) -> ToolRunRequest:
    return ToolRunRequest(ToolName("edit_file"), ToolInput(payload))


def edit_payload(path: Path, old: str, new: str) -> str:
    return json.dumps({"path": str(path), "old": old, "new": new})


class TestEditFileTool:
    def test_advertises_a_sensitive_spec(self) -> None:
        # Act
        spec = EditFileTool().spec

        # Assert
        assert spec.name == ToolName("edit_file")
        assert spec.requires_approval is True

    def test_replaces_a_unique_occurrence(self, tmp_path: Path) -> None:
        # Arrange
        target = tmp_path / "app.py"
        target.write_text("x = 1\ny = 2\n", encoding="utf-8")

        # Act
        result = EditFileTool().run(
            edit_request(edit_payload(target, "x = 1", "x = 99"))
        )

        # Assert
        assert result.tool_name == ToolName("edit_file")
        assert result.succeeded is True
        assert target.read_text(encoding="utf-8") == "x = 99\ny = 2\n"
        assert result.output.value == f"Replaced 1 occurrence in {target}."

    def test_reports_a_failure_when_the_text_is_not_found(self, tmp_path: Path) -> None:
        # Arrange
        target = tmp_path / "app.py"
        target.write_text("x = 1\n", encoding="utf-8")

        # Act
        result = EditFileTool().run(edit_request(edit_payload(target, "absent", "z")))

        # Assert: the exact message names the path, the snippet, and the rule.
        assert result.succeeded is False
        assert result.output.value == (
            f"Edit error: Text to replace not found in {target}: 'absent'. "
            "Expected it to occur exactly once."
        )

    def test_reports_a_failure_when_the_text_is_ambiguous(self, tmp_path: Path) -> None:
        # Arrange: two identical lines make a single replacement ambiguous.
        target = tmp_path / "app.py"
        target.write_text("dup\ndup\n", encoding="utf-8")

        # Act
        result = EditFileTool().run(edit_request(edit_payload(target, "dup", "z")))

        # Assert: the exact message reports the count and the uniqueness rule.
        assert result.succeeded is False
        assert result.output.value == (
            f"Edit error: Text to replace occurs 2 times in {target}: 'dup'. "
            "Expected exactly one occurrence."
        )

    def test_reports_a_failure_when_the_file_is_missing(self, tmp_path: Path) -> None:
        # Act: reading a missing file raises OSError, handled as a failure.
        missing = tmp_path / "absent.py"
        result = EditFileTool().run(edit_request(edit_payload(missing, "a", "b")))

        # Assert
        assert result.succeeded is False
        assert result.output.value.startswith("Edit error:")

    def test_reports_a_failure_for_invalid_json(self) -> None:
        # Act
        result = EditFileTool().run(edit_request("not json"))

        # Assert
        assert result.tool_name == ToolName("edit_file")
        assert result.succeeded is False
        assert result.output.value.startswith("Edit error:")
