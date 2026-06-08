"""Unit tests for PythonGrepSearch.

TDD: tests are written before the implementation. They define the contract
for how PythonGrepSearch behaves: exit codes, output format, flag handling,
binary-file skipping, and timeout signalling.

The old tests drove subprocess mechanics using POSIX binaries (echo, false,
ls, sleep). These tests drive filesystem behaviour directly using tmp_path.
"""

from __future__ import annotations

import time
from pathlib import Path

from little_harness_ripgrep.ripgrep_search import PythonGrepSearch

MATCH_EXIT_CODE = 0
NO_MATCH_EXIT_CODE = 1
SEARCH_ERROR_EXIT_CODE = 2

_search = PythonGrepSearch()


def _write(tmp_path: Path, name: str, content: str) -> Path:
    """Write text content to a file in tmp_path and return its path."""
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


class TestPythonGrepSearchMatches:
    def test_finds_a_match_and_returns_exit_code_zero(self, tmp_path: Path) -> None:
        _write(tmp_path, "app.py", "# TODO: refactor this\n")

        outcome = _search.run(["TODO", str(tmp_path)], 30.0)

        assert outcome.exit_code == MATCH_EXIT_CODE
        assert "TODO" in outcome.stdout
        assert outcome.stderr == ""

    def test_output_is_filename_colon_lineno_colon_content(
        self, tmp_path: Path
    ) -> None:
        path = _write(tmp_path, "notes.txt", "first line\nTODO here\nthird line\n")

        outcome = _search.run(["TODO", str(tmp_path)], 30.0)

        assert outcome.exit_code == MATCH_EXIT_CODE
        # Format: path:line_number:matched_content
        assert f"{path}:2:TODO here" in outcome.stdout

    def test_returns_all_matches_across_multiple_files(self, tmp_path: Path) -> None:
        _write(tmp_path, "a.py", "TODO in a\n")
        _write(tmp_path, "b.py", "TODO in b\n")

        outcome = _search.run(["TODO", str(tmp_path)], 30.0)

        assert outcome.exit_code == MATCH_EXIT_CODE
        assert "a.py" in outcome.stdout
        assert "b.py" in outcome.stdout

    def test_returns_exit_one_when_no_match_found(self, tmp_path: Path) -> None:
        _write(tmp_path, "clean.py", "no issues here\n")

        outcome = _search.run(["MISSING_PATTERN_XYZ", str(tmp_path)], 30.0)

        assert outcome.exit_code == NO_MATCH_EXIT_CODE
        assert outcome.stdout == ""
        assert outcome.stderr == ""

    def test_returns_exit_one_on_empty_directory(self, tmp_path: Path) -> None:
        outcome = _search.run(["anything", str(tmp_path)], 30.0)

        assert outcome.exit_code == NO_MATCH_EXIT_CODE

    def test_searches_a_single_file_directly(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "single.py", "match here\n")

        outcome = _search.run(["match", str(path)], 30.0)

        assert outcome.exit_code == MATCH_EXIT_CODE
        assert "match" in outcome.stdout

    def test_searches_nested_subdirectories_recursively(self, tmp_path: Path) -> None:
        sub = tmp_path / "sub" / "deep"
        sub.mkdir(parents=True)
        (sub / "nested.py").write_text("TODO: nested\n", encoding="utf-8")

        outcome = _search.run(["TODO", str(tmp_path)], 30.0)

        assert outcome.exit_code == MATCH_EXIT_CODE
        assert "nested.py" in outcome.stdout


class TestPythonGrepSearchFlags:
    def test_ignorecase_flag_matches_regardless_of_case(self, tmp_path: Path) -> None:
        _write(tmp_path, "app.py", "todo: fix this\n")

        outcome = _search.run(["-i", "TODO", str(tmp_path)], 30.0)

        assert outcome.exit_code == MATCH_EXIT_CODE
        assert "todo" in outcome.stdout.lower()

    def test_without_ignorecase_uppercase_pattern_does_not_match_lowercase(
        self, tmp_path: Path
    ) -> None:
        _write(tmp_path, "app.py", "todo: fix this\n")

        outcome = _search.run(["TODO", str(tmp_path)], 30.0)

        assert outcome.exit_code == NO_MATCH_EXIT_CODE

    def test_type_flag_restricts_to_python_files(self, tmp_path: Path) -> None:
        _write(tmp_path, "match.py", "TODO in py file\n")
        _write(tmp_path, "no_match.txt", "TODO in txt file\n")

        outcome = _search.run(["-t", "py", "TODO", str(tmp_path)], 30.0)

        assert outcome.exit_code == MATCH_EXIT_CODE
        assert "match.py" in outcome.stdout
        assert "no_match.txt" not in outcome.stdout

    def test_max_count_caps_matches_per_file(self, tmp_path: Path) -> None:
        _write(
            tmp_path,
            "many.py",
            "TODO line 1\nTODO line 2\nTODO line 3\nTODO line 4\n",
        )

        outcome = _search.run(["-m", "2", "TODO", str(tmp_path)], 30.0)

        assert outcome.exit_code == MATCH_EXIT_CODE
        matched_lines = [line for line in outcome.stdout.splitlines() if "TODO" in line]
        assert len(matched_lines) <= 2  # noqa: PLR2004


class TestPythonGrepSearchSafety:
    def test_binary_file_is_skipped_silently(self, tmp_path: Path) -> None:
        binary_path = tmp_path / "binary.bin"
        # Null bytes make a file binary; embed pattern alongside them.
        binary_path.write_bytes(b"TODO\x00\x00\x00binary data")

        outcome = _search.run(["TODO", str(tmp_path)], 30.0)

        # Binary file must not appear in results.
        assert "binary.bin" not in outcome.stdout

    def test_hidden_directories_are_skipped(self, tmp_path: Path) -> None:
        hidden = tmp_path / ".hidden_dir"
        hidden.mkdir()
        (hidden / "secret.py").write_text("TODO secret\n", encoding="utf-8")

        outcome = _search.run(["TODO", str(tmp_path)], 30.0)

        assert ".hidden_dir" not in outcome.stdout

    def test_git_directory_is_skipped(self, tmp_path: Path) -> None:
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "COMMIT_EDITMSG").write_text("TODO commit\n", encoding="utf-8")

        outcome = _search.run(["TODO", str(tmp_path)], 30.0)

        assert ".git" not in outcome.stdout

    def test_pycache_directory_is_skipped(self, tmp_path: Path) -> None:
        cache = tmp_path / "__pycache__"
        cache.mkdir()
        (cache / "cached.pyc").write_text("TODO cached\n", encoding="utf-8")

        outcome = _search.run(["TODO", str(tmp_path)], 30.0)

        assert "__pycache__" not in outcome.stdout

    def test_timeout_does_not_crash_and_returns_an_outcome(
        self, tmp_path: Path
    ) -> None:
        # Create enough files that we might hit the timeout.
        for i in range(50):
            _write(tmp_path, f"file_{i}.py", "TODO\n" * 100)

        start = time.monotonic()
        outcome = _search.run(["TODO", str(tmp_path)], 0.05)
        elapsed = time.monotonic() - start

        # Must return within a reasonable margin of the timeout.
        assert elapsed < 5.0  # noqa: PLR2004
        # Must return a valid outcome (not crash).
        assert outcome.exit_code in (MATCH_EXIT_CODE, NO_MATCH_EXIT_CODE)


class TestPythonGrepSearchErrors:
    def test_invalid_regex_returns_exit_code_two(self, tmp_path: Path) -> None:
        outcome = _search.run(["(unclosed", str(tmp_path)], 30.0)

        assert outcome.exit_code == SEARCH_ERROR_EXIT_CODE
        assert outcome.stdout == ""
        assert "(unclosed" in outcome.stderr

    def test_nonexistent_path_returns_exit_code_two(self) -> None:
        outcome = _search.run(["TODO", "/no/such/path/xyz_abc_123"], 30.0)

        assert outcome.exit_code == SEARCH_ERROR_EXIT_CODE
        assert outcome.stdout == ""
        assert "/no/such/path/xyz_abc_123" in outcome.stderr

    def test_unknown_flag_returns_exit_code_two(self, tmp_path: Path) -> None:
        outcome = _search.run(["--no-such-flag", "TODO", str(tmp_path)], 30.0)

        assert outcome.exit_code == SEARCH_ERROR_EXIT_CODE
        assert outcome.stdout == ""

    def test_unknown_type_returns_exit_code_two(self, tmp_path: Path) -> None:
        outcome = _search.run(["-t", "cobol", "TODO", str(tmp_path)], 30.0)

        assert outcome.exit_code == SEARCH_ERROR_EXIT_CODE
        assert "cobol" in outcome.stderr
