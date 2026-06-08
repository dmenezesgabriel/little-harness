"""Unit tests for PythonGrepSearch.

TDD: tests are written before the implementation. They define the contract
for how PythonGrepSearch behaves: exit codes, output format, flag handling,
binary-file skipping, and timeout signalling.

The old tests drove subprocess mechanics using POSIX binaries (echo, false,
ls, sleep). These tests drive filesystem behaviour directly using tmp_path.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import patch

from little_harness_ripgrep.ripgrep_search import GrepArgumentParser, PythonGrepSearch

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
        path = _write(tmp_path, "app.py", "# TODO: refactor this\n")

        outcome = _search.run(["TODO", str(tmp_path)], 30.0)

        assert outcome.exit_code == MATCH_EXIT_CODE
        assert outcome.stdout == f"{path}:1:# TODO: refactor this\n"
        assert outcome.stderr == ""

    def test_output_is_filename_colon_lineno_colon_content(
        self, tmp_path: Path
    ) -> None:
        path = _write(tmp_path, "notes.txt", "first line\nTODO here\nthird line\n")

        outcome = _search.run(["TODO", str(tmp_path)], 30.0)

        assert outcome.exit_code == MATCH_EXIT_CODE
        assert outcome.stdout == f"{path}:2:TODO here\n"

    def test_preserves_trailing_whitespace_on_match(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "space.txt", "TODO match   \n")

        outcome = _search.run(["TODO", str(tmp_path)], 30.0)

        assert outcome.exit_code == MATCH_EXIT_CODE
        assert outcome.stdout == f"{path}:1:TODO match   \n"

    def test_preserves_trailing_char_x_on_match(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "char_x.txt", "TODO match X\n")

        outcome = _search.run(["TODO", str(tmp_path)], 30.0)

        assert outcome.exit_code == MATCH_EXIT_CODE
        assert outcome.stdout == f"{path}:1:TODO match X\n"

    def test_returns_all_matches_across_multiple_files(self, tmp_path: Path) -> None:
        file_a = _write(tmp_path, "a.py", "TODO in a\n")
        file_b = _write(tmp_path, "b.py", "TODO in b\n")

        outcome = _search.run(["TODO", str(tmp_path)], 30.0)

        assert outcome.exit_code == MATCH_EXIT_CODE
        expected = sorted([f"{file_a}:1:TODO in a", f"{file_b}:1:TODO in b"])
        assert sorted(outcome.stdout.splitlines()) == expected

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
        assert outcome.stdout == f"{path}:1:match here\n"

    def test_searches_nested_subdirectories_recursively(self, tmp_path: Path) -> None:
        sub = tmp_path / "sub" / "deep"
        sub.mkdir(parents=True)
        path = sub / "nested.py"
        path.write_text("TODO: nested\n", encoding="utf-8")

        outcome = _search.run(["TODO", str(tmp_path)], 30.0)

        assert outcome.exit_code == MATCH_EXIT_CODE
        assert outcome.stdout == f"{path}:1:TODO: nested\n"


class TestPythonGrepSearchFlags:
    def test_ignorecase_flag_matches_regardless_of_case(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "app.py", "todo: fix this\n")

        outcome = _search.run(["-i", "TODO", str(tmp_path)], 30.0)

        assert outcome.exit_code == MATCH_EXIT_CODE
        assert outcome.stdout == f"{path}:1:todo: fix this\n"

    def test_without_ignorecase_uppercase_pattern_does_not_match_lowercase(
        self, tmp_path: Path
    ) -> None:
        _write(tmp_path, "app.py", "todo: fix this\n")

        outcome = _search.run(["TODO", str(tmp_path)], 30.0)

        assert outcome.exit_code == NO_MATCH_EXIT_CODE

    def test_type_flag_restricts_to_python_files(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "match.py", "TODO in py file\n")
        _write(tmp_path, "no_match.txt", "TODO in txt file\n")

        outcome = _search.run(["-t", "py", "TODO", str(tmp_path)], 30.0)

        assert outcome.exit_code == MATCH_EXIT_CODE
        assert outcome.stdout == f"{path}:1:TODO in py file\n"


class TestPythonGrepSearchSafety:
    def test_binary_file_is_skipped_silently(self, tmp_path: Path) -> None:
        binary_path = tmp_path / "binary.bin"
        binary_path.write_bytes(b"TODO\x00\x00\x00binary data")

        outcome = _search.run(["TODO", str(tmp_path)], 30.0)

        assert "binary.bin" not in outcome.stdout

    def test_binary_file_detection_boundary_skipped_at_1024(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "boundary_skip.bin"
        content = b"TODO match" + b" " * 1013 + b"\x00"
        path.write_bytes(content)

        outcome = _search.run(["TODO", str(tmp_path)], 30.0)

        assert outcome.exit_code == NO_MATCH_EXIT_CODE
        assert "boundary_skip.bin" not in outcome.stdout

    def test_binary_file_detection_boundary_matched_at_1025(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "boundary_match.bin"
        content = b"TODO match" + b" " * 1014 + b"\x00"
        path.write_bytes(content)

        outcome = _search.run(["TODO", str(tmp_path)], 30.0)

        assert outcome.exit_code == MATCH_EXIT_CODE
        assert outcome.stdout == f"{path}:1:TODO match" + " " * 1014 + "\x00\n"

    def test_is_binary_handles_oserror_by_skipping(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "visible.py", "TODO\n")
        original_open: Callable[..., Any] = Path.open
        open_count = 0

        def mock_open(self_path: Path, *args: Any, **kwargs: Any) -> Any:
            nonlocal open_count
            if self_path == path:
                open_count += 1
                raise OSError("Permission denied")
            return original_open(self_path, *args, **kwargs)

        with patch.object(Path, "open", mock_open):
            outcome = _search.run(["TODO", str(path)], 30.0)

        assert outcome.exit_code == NO_MATCH_EXIT_CODE
        assert open_count == 1

    def test_grep_file_explicitly_uses_utf8_encoding(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "file.py", "TODO\n")
        original_open: Callable[..., Any] = Path.open
        open_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

        def mock_open(self_path: Path, *args: Any, **kwargs: Any) -> Any:
            open_calls.append((args, kwargs))
            return original_open(self_path, *args, **kwargs)

        with patch.object(Path, "open", mock_open):
            _search.run(["TODO", str(path)], 30.0)

        # The second open call is inside _grep_file.
        # It must explicitly use mode="r", encoding="utf-8", and errors="replace".
        assert open_calls[1] == (("r",), {"encoding": "utf-8", "errors": "replace"})

    def test_invalid_utf8_is_successfully_replaced_and_matched(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "invalid_utf8.py"
        path.write_bytes(b"TODO match \xff\n")

        outcome = _search.run(["TODO", str(tmp_path)], 30.0)

        assert outcome.exit_code == MATCH_EXIT_CODE
        assert outcome.stdout == f"{path}:1:TODO match \ufffd\n"

    def test_hidden_directories_are_skipped(self, tmp_path: Path) -> None:
        hidden = tmp_path / ".hidden_dir"
        hidden.mkdir()
        (hidden / "secret.py").write_text("TODO secret\n", encoding="utf-8")

        outcome = _search.run(["TODO", str(tmp_path)], 30.0)

        assert ".hidden_dir" not in outcome.stdout

    def test_hidden_files_are_skipped(self, tmp_path: Path) -> None:
        file_a = _write(tmp_path, "a.py", "TODO visible a\n")
        _write(tmp_path, ".hidden.py", "TODO hidden\n")
        file_z = _write(tmp_path, "z.py", "TODO visible z\n")

        outcome = _search.run(["TODO", str(tmp_path)], 30.0)

        assert outcome.exit_code == MATCH_EXIT_CODE
        expected = sorted([f"{file_a}:1:TODO visible a", f"{file_z}:1:TODO visible z"])
        assert sorted(outcome.stdout.splitlines()) == expected

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

    def test_timeout_stops_execution_and_returns_partial_results(
        self, tmp_path: Path
    ) -> None:
        file1 = _write(tmp_path, "file1.py", "TODO match 1\n")
        file2 = _write(tmp_path, "file2.py", "TODO match 2\n")

        with patch("time.monotonic", side_effect=[10.0, 10.1] + [20.0] * 50):
            outcome = _search.run(["TODO", str(tmp_path)], 1.0)

        assert outcome.exit_code == MATCH_EXIT_CODE
        assert outcome.stdout in (
            f"{file1}:1:TODO match 1\n",
            f"{file2}:1:TODO match 2\n",
        )

    def test_timeout_between_multiple_paths(self, tmp_path: Path) -> None:
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        file1 = _write(dir_a, "file1.py", "TODO match 1\n")
        _write(dir_b, "file2.py", "TODO match 2\n")

        with patch("time.monotonic", side_effect=[10.0, 10.1] + [20.0] * 50):
            outcome = _search.run(["TODO", str(dir_a), str(dir_b)], 1.0)

        assert outcome.exit_code == MATCH_EXIT_CODE
        assert outcome.stdout == f"{file1}:1:TODO match 1\n"

    def test_timeout_exact_deadline_not_triggered(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "file.py", "TODO\n")

        with patch("time.monotonic", side_effect=[10.0, 11.0] + [20.0] * 50):
            outcome = _search.run(["TODO", str(tmp_path)], 1.0)

        assert outcome.exit_code == MATCH_EXIT_CODE
        assert outcome.stdout == f"{path}:1:TODO\n"

    def test_timeout_between_paths_exact_deadline_not_triggered(
        self, tmp_path: Path
    ) -> None:
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        file1 = _write(dir_a, "file1.py", "TODO match 1\n")
        file2 = _write(dir_b, "file2.py", "TODO match 2\n")

        # Call 1 (start): 10.0. deadline is 11.0.
        # Call 2 (in loop): 10.1 (not timed out).
        # Call 3 (outside loop check): 11.0 (exact deadline). Should NOT time out.
        # Call 4 (in loop for dir_b): 11.0 (exact deadline). Should NOT time out.
        with patch(
            "time.monotonic", side_effect=[10.0, 10.1, 11.0, 11.0] + [20.0] * 50
        ):
            outcome = _search.run(["TODO", str(dir_a), str(dir_b)], 1.0)

        assert outcome.exit_code == MATCH_EXIT_CODE
        expected = sorted([f"{file1}:1:TODO match 1", f"{file2}:1:TODO match 2"])
        assert sorted(outcome.stdout.splitlines()) == expected

    def test_max_count_caps_matches_per_file(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            "many.py",
            "TODO line 1\nTODO line 2\nTODO line 3\nTODO line 4\n",
        )

        outcome = _search.run(["-m", "2", "TODO", str(tmp_path)], 30.0)

        assert outcome.exit_code == MATCH_EXIT_CODE
        assert outcome.stdout == f"{path}:1:TODO line 1\n{path}:2:TODO line 2\n"

    def test_pythongrepsearch_init_with_custom_parser(self) -> None:
        custom_parser = GrepArgumentParser()
        search = PythonGrepSearch(parser=custom_parser)
        assert search._parser is custom_parser  # pyright: ignore[reportPrivateUsage]

    def test_pythongrepsearch_init_with_default_parser(self) -> None:
        search = PythonGrepSearch()
        assert isinstance(search._parser, GrepArgumentParser)  # pyright: ignore[reportPrivateUsage]


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
        assert "--no-such-flag" in outcome.stderr

    def test_unknown_type_returns_exit_code_two(self, tmp_path: Path) -> None:
        outcome = _search.run(["-t", "cobol", "TODO", str(tmp_path)], 30.0)

        assert outcome.exit_code == SEARCH_ERROR_EXIT_CODE
        assert outcome.stdout == ""
        assert "cobol" in outcome.stderr
