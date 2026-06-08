"""Unit tests for GrepArgumentParser.

TDD: these tests are written before the implementation exists.
They define the contract for how the parser maps a ripgrep-style argument
list into a typed GrepRequest, and how it signals errors via RipgrepOutcome.
"""

from __future__ import annotations

import re
from pathlib import Path

from little_harness_ripgrep.ripgrep_search import (
    GrepArgumentParser,
    GrepRequest,
    RipgrepOutcome,
    _GrepParseError,  # pyright: ignore[reportPrivateUsage]
)

_EXIT_ERROR = 2

_parser = GrepArgumentParser()


def parse(arguments: list[str], tmp_path: Path) -> GrepRequest | RipgrepOutcome:
    # Replace placeholder "." with a real path so existence checks pass.
    resolved = [str(tmp_path) if a == "." else a for a in arguments]
    return _parser.parse(resolved)


class TestGrepArgumentParserSuccess:
    def test_parses_pattern_and_defaults_path_to_cwd(self, tmp_path: Path) -> None:
        result = parse(["TODO", str(tmp_path)], tmp_path)

        assert isinstance(result, GrepRequest)
        assert result.pattern.pattern == "TODO"
        assert result.paths == (tmp_path,)

    def test_compiled_pattern_is_case_sensitive_by_default(
        self, tmp_path: Path
    ) -> None:
        result = parse(["TODO", str(tmp_path)], tmp_path)

        assert isinstance(result, GrepRequest)
        assert result.pattern.flags == re.UNICODE

    def test_ignorecase_flag_compiles_case_insensitive(self, tmp_path: Path) -> None:
        result = parse(["-i", "TODO", str(tmp_path)], tmp_path)

        assert isinstance(result, GrepRequest)
        assert result.pattern.flags == re.UNICODE | re.IGNORECASE

    def test_long_ignorecase_flag_also_works(self, tmp_path: Path) -> None:
        result = parse(["--ignore-case", "TODO", str(tmp_path)], tmp_path)

        assert isinstance(result, GrepRequest)
        assert result.pattern.flags == re.UNICODE | re.IGNORECASE

    def test_type_flag_sets_file_extensions(self, tmp_path: Path) -> None:
        result = parse(["-t", "py", "TODO", str(tmp_path)], tmp_path)

        assert isinstance(result, GrepRequest)
        assert result.file_extensions == frozenset({".py"})
        assert result.pattern.pattern == "TODO"
        assert result.paths == (tmp_path,)

    def test_type_flag_at_end_of_arguments(self, tmp_path: Path) -> None:
        result = parse(["TODO", "-t", "py"], tmp_path)

        assert isinstance(result, GrepRequest)
        assert result.file_extensions == frozenset({".py"})
        assert result.pattern.pattern == "TODO"
        assert result.paths == (Path(),)

    def test_no_type_flag_leaves_extensions_as_none(self, tmp_path: Path) -> None:
        result = parse(["TODO", str(tmp_path)], tmp_path)

        assert isinstance(result, GrepRequest)
        assert result.file_extensions is None

    def test_max_count_flag_sets_per_file_limit(self, tmp_path: Path) -> None:
        result = parse(["-m", "5", "TODO", str(tmp_path)], tmp_path)

        assert isinstance(result, GrepRequest)
        assert result.max_results_per_file == 5  # noqa: PLR2004
        assert result.pattern.pattern == "TODO"
        assert result.paths == (tmp_path,)

    def test_max_count_flag_at_end_of_arguments(self, tmp_path: Path) -> None:
        result = parse(["TODO", "-m", "5"], tmp_path)

        assert isinstance(result, GrepRequest)
        assert result.max_results_per_file == 5  # noqa: PLR2004
        assert result.pattern.pattern == "TODO"
        assert result.paths == (Path(),)

    def test_long_max_count_flag_also_works(self, tmp_path: Path) -> None:
        result = parse(["--max-count", "3", "TODO", str(tmp_path)], tmp_path)

        assert isinstance(result, GrepRequest)
        assert result.max_results_per_file == 3  # noqa: PLR2004
        assert result.pattern.pattern == "TODO"
        assert result.paths == (tmp_path,)

    def test_no_max_count_leaves_limit_as_none(self, tmp_path: Path) -> None:
        result = parse(["TODO", str(tmp_path)], tmp_path)

        assert isinstance(result, GrepRequest)
        assert result.max_results_per_file is None

    def test_multiple_paths_are_accepted(self, tmp_path: Path) -> None:
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()

        result = _parser.parse(["TODO", str(dir_a), str(dir_b)])

        assert isinstance(result, GrepRequest)
        assert result.paths == (dir_a, dir_b)

    def test_double_dash_stops_flag_parsing(self, tmp_path: Path) -> None:
        result = parse(["--", "-i", str(tmp_path)], tmp_path)

        assert isinstance(result, GrepRequest)
        assert result.pattern.pattern == "-i"
        assert result.pattern.flags == re.UNICODE

    def test_single_dash_treated_as_positional(self, tmp_path: Path) -> None:
        result = parse(["-", str(tmp_path)], tmp_path)

        assert isinstance(result, GrepRequest)
        assert result.pattern.pattern == "-"


class TestGrepArgumentParserErrors:
    def test_invalid_regex_returns_exit_code_2(self, tmp_path: Path) -> None:
        result = parse(["(unclosed", str(tmp_path)], tmp_path)

        assert isinstance(result, RipgrepOutcome)
        assert result.exit_code == _EXIT_ERROR
        assert result.stdout == ""
        assert "(unclosed" in result.stderr

    def test_nonexistent_path_returns_exit_code_2(self) -> None:
        result = _parser.parse(["TODO", "/no/such/path/xyz123"])

        assert isinstance(result, RipgrepOutcome)
        assert result.exit_code == _EXIT_ERROR
        assert result.stdout == ""
        assert "/no/such/path/xyz123" in result.stderr

    def test_unknown_type_returns_exit_code_2(self, tmp_path: Path) -> None:
        result = parse(["-t", "cobol", "TODO", str(tmp_path)], tmp_path)

        assert isinstance(result, RipgrepOutcome)
        assert result.exit_code == _EXIT_ERROR
        assert result.stdout == ""
        assert "cobol" in result.stderr

    def test_unknown_flag_returns_exit_code_2(self, tmp_path: Path) -> None:
        result = parse(["--no-such-flag", "TODO", str(tmp_path)], tmp_path)

        assert isinstance(result, RipgrepOutcome)
        assert result.exit_code == _EXIT_ERROR
        assert result.stdout == ""
        assert result.stderr == "Error: unknown flag --no-such-flag"

    def test_missing_type_argument_short(self, tmp_path: Path) -> None:
        result = parse(["-t"], tmp_path)

        assert isinstance(result, RipgrepOutcome)
        assert result.exit_code == _EXIT_ERROR
        assert result.stdout == ""
        assert result.stderr == "Error: missing argument for -t/--type flag"

    def test_missing_type_argument_long(self, tmp_path: Path) -> None:
        result = parse(["--type"], tmp_path)

        assert isinstance(result, RipgrepOutcome)
        assert result.exit_code == _EXIT_ERROR
        assert result.stdout == ""
        assert result.stderr == "Error: missing argument for -t/--type flag"

    def test_missing_max_count_argument_short(self, tmp_path: Path) -> None:
        result = parse(["-m"], tmp_path)

        assert isinstance(result, RipgrepOutcome)
        assert result.exit_code == _EXIT_ERROR
        assert result.stdout == ""
        assert result.stderr == "Error: missing argument for -m/--max-count flag"

    def test_missing_max_count_argument_long(self, tmp_path: Path) -> None:
        result = parse(["--max-count"], tmp_path)

        assert isinstance(result, RipgrepOutcome)
        assert result.exit_code == _EXIT_ERROR
        assert result.stdout == ""
        assert result.stderr == "Error: missing argument for -m/--max-count flag"

    def test_invalid_max_count_value(self, tmp_path: Path) -> None:
        result = parse(["-m", "not-an-int", "TODO", str(tmp_path)], tmp_path)

        assert isinstance(result, RipgrepOutcome)
        assert result.exit_code == _EXIT_ERROR
        assert result.stdout == ""
        assert result.stderr == "Error: invalid max-count value 'not-an-int'"

    def test_missing_search_pattern(self, tmp_path: Path) -> None:
        result = parse([], tmp_path)

        assert isinstance(result, RipgrepOutcome)
        assert result.exit_code == _EXIT_ERROR
        assert result.stdout == ""
        assert result.stderr == "Error: missing search pattern"


def test_grep_parse_error_exception_message() -> None:
    outcome = RipgrepOutcome(exit_code=2, stdout="", stderr="some error message")
    err = _GrepParseError(outcome)
    assert str(err) == "some error message"
