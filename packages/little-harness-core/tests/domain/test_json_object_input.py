from __future__ import annotations

import pytest
from little_harness.domain.json_object_input import JsonObjectInput


class TestJsonObjectInput:
    def test_reads_required_string_fields(self) -> None:
        # Act
        fields = JsonObjectInput.parse('{"path": "a.txt", "content": "hi"}')

        # Assert
        assert fields.required_text("path") == "a.txt"
        assert fields.required_text("content") == "hi"

    def test_rejects_input_that_is_not_valid_json(self) -> None:
        # Act / Assert
        with pytest.raises(ValueError, match=r"Invalid JSON tool input: 'not json'"):
            JsonObjectInput.parse("not json")

    def test_rejects_json_that_is_not_an_object(self) -> None:
        # Act / Assert: a JSON array is valid JSON but the wrong shape.
        with pytest.raises(ValueError, match=r"Expected a JSON object, got list"):
            JsonObjectInput.parse("[1, 2]")

    def test_rejects_a_missing_required_field(self) -> None:
        # Arrange
        fields = JsonObjectInput.parse('{"path": "a.txt"}')

        # Act / Assert
        with pytest.raises(ValueError, match=r"Missing field 'content'"):
            fields.required_text("content")

    def test_rejects_a_field_that_is_not_a_string(self) -> None:
        # Arrange
        fields = JsonObjectInput.parse('{"path": 7}')

        # Act / Assert
        with pytest.raises(ValueError, match=r"Field 'path' must be a string, got 7"):
            fields.required_text("path")
