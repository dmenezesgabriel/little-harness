# ruff: noqa: D100, D101, D102, D103
import json
from pathlib import Path


from little_harness_session_jsonl.infrastructure.jsonl_appender import JsonlFileAppender


class TestJsonlFileAppender:
    def test_appends_valid_json_line_with_newline(self, tmp_path: Path) -> None:
        # Arrange
        file_path = tmp_path / "test.jsonl"
        appender = JsonlFileAppender(file_path)
        data = {"event": "run_started", "run_id": "123"}

        # Act
        appender.append(data)

        # Assert
        assert file_path.exists()
        content = file_path.read_text()
        assert content.endswith("\n")
        parsed = json.loads(content)
        assert parsed == data

    def test_appends_multiple_lines_atomically(self, tmp_path: Path) -> None:
        # Arrange
        file_path = tmp_path / "nested" / "deep" / "multiple.jsonl"
        appender = JsonlFileAppender(file_path)

        # Act
        appender.append({"id": 1, "name": "test"})
        appender.append({"id": 2, "name": "test2"})

        # Assert
        assert file_path.parent.exists()

        content = file_path.read_text(encoding="utf-8")
        lines = content.splitlines()

        # Exact string assertions to kill separator mutants
        assert lines[0] == '{"id":1,"name":"test"}'
        assert lines[1] == '{"id":2,"name":"test2"}'
        assert content.endswith("\n")
        assert len(lines) == 2
