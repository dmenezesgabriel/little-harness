"""Exercises the real py-tree-sitter adapter and the language guard."""

from __future__ import annotations

import pytest
from little_harness_ast.tree_sitter_engine import (
    LANGUAGE_FACTORIES,
    TreeSitterEngine,
    load_language,
)

TWO_CALLS = "print('hi')\nprint('bye')\n"
EXPECTED_CALL_COUNT = 2
FIRST_END_BYTE = 11
SECOND_START_BYTE = 12
SECOND_END_BYTE = 24
SECOND_LINE = 2

EXPECTED_LANGUAGES = frozenset({"python", "javascript"})


class TestTreeSitterEngine:
    def test_finds_each_match_with_lines_bytes_and_text(self) -> None:
        # Act
        matches = TreeSitterEngine().find_matches(TWO_CALLS, "python", "(call) @match")

        # Assert: two calls, located by line and byte span, in document order.
        assert len(matches) == EXPECTED_CALL_COUNT
        assert matches[0].start_line == 1
        assert matches[0].start_byte == 0
        assert matches[0].end_byte == FIRST_END_BYTE
        assert matches[0].text == "print('hi')"
        assert matches[1].start_line == SECOND_LINE
        assert matches[1].start_byte == SECOND_START_BYTE
        assert matches[1].end_byte == SECOND_END_BYTE

    def test_reports_a_multi_line_span(self) -> None:
        # Act
        matches = TreeSitterEngine().find_matches(
            "def f():\n    return 1\n", "python", "(function_definition) @match"
        )

        # Assert: start and end lines are reported independently.
        assert len(matches) == 1
        assert matches[0].start_line == 1
        assert matches[0].end_line == SECOND_LINE

    def test_searches_a_non_python_language(self) -> None:
        # Act
        matches = TreeSitterEngine().find_matches(
            "console.log(1)", "javascript", "(call_expression) @match"
        )

        # Assert
        assert len(matches) == 1

    def test_returns_nothing_when_the_query_has_no_match_capture(self) -> None:
        # Act: a query without an @match capture yields no targets.
        matches = TreeSitterEngine().find_matches(TWO_CALLS, "python", "(call) @other")

        # Assert
        assert list(matches) == []

    def test_rejects_an_unsupported_language(self) -> None:
        # Act / Assert
        with pytest.raises(ValueError, match=r"Unsupported language: 'cobol'"):
            TreeSitterEngine().find_matches("x", "cobol", "(x) @match")

    def test_rejects_an_invalid_query(self) -> None:
        # Act / Assert: a malformed query is a clean error, not a vendor type.
        with pytest.raises(ValueError, match=r"Invalid query: '\(\(\('"):
            TreeSitterEngine().find_matches("x = 1", "python", "(((")


class TestLoadLanguage:
    def test_loads_a_supported_language(self) -> None:
        # Act / Assert: a known language returns a usable Language object.
        assert load_language("python") is not None

    def test_rejects_an_unsupported_language(self) -> None:
        # Act / Assert
        with pytest.raises(ValueError, match=r"Unsupported language: 'fortran'"):
            load_language("fortran")

    def test_supports_exactly_the_expected_languages(self) -> None:
        # Pinning the set guards against silently adding/removing a language.
        assert frozenset(LANGUAGE_FACTORIES) == EXPECTED_LANGUAGES
