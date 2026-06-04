"""Exercises the real subprocess boundary of the ripgrep search.

The boundary is binary-agnostic: it runs an injected binary and captures the
outcome. ripgrep itself may be absent (and is a shell function here), so these
tests drive the mechanics with portable POSIX binaries whose exit codes mirror
ripgrep's (0 = match, 1 = no match, 2 = error). ripgrep's exit-code *meaning* is
interpreted in `RipgrepTool`, tested separately with fakes.
"""

from __future__ import annotations

from little_harness_ripgrep.ripgrep_search import (
    RIPGREP_BINARY,
    SubprocessRipgrepSearch,
)

MATCH_EXIT_CODE = 0
NO_MATCH_EXIT_CODE = 1
SEARCH_ERROR_EXIT_CODE = 2


class TestSubprocessRipgrepSearch:
    def test_defaults_to_the_ripgrep_binary(self) -> None:
        # The default binary is ripgrep itself; stand-ins are for tests only.
        assert RIPGREP_BINARY == "rg"

    def test_captures_stdout_and_a_zero_exit_code(self) -> None:
        # Act: `echo` stands in for a successful, output-producing search.
        outcome = SubprocessRipgrepSearch("echo").run(["a-match"], 5.0)

        # Assert
        assert outcome.exit_code == MATCH_EXIT_CODE
        assert outcome.stdout == "a-match\n"
        assert outcome.stderr == ""

    def test_captures_exit_code_one_with_no_output(self) -> None:
        # Act: `false` stands in for ripgrep's "no matches" (exit 1).
        outcome = SubprocessRipgrepSearch("false").run([], 5.0)

        # Assert
        assert outcome.exit_code == NO_MATCH_EXIT_CODE
        assert outcome.stdout == ""

    def test_captures_stderr_and_an_error_exit_code(self) -> None:
        # Act: `ls` on a missing path stands in for a search error (exit 2).
        outcome = SubprocessRipgrepSearch("ls").run(["/no-such-path-xyz-12345"], 5.0)

        # Assert
        assert outcome.exit_code == SEARCH_ERROR_EXIT_CODE
        assert outcome.stderr != ""

    def test_reports_an_absent_binary(self) -> None:
        # Act
        outcome = SubprocessRipgrepSearch("rg-not-real-xyz").run(["needle"], 5.0)

        # Assert: a missing binary is a None exit code with an actionable message.
        assert outcome.exit_code is None
        assert outcome.stdout == ""
        assert outcome.stderr == (
            "ripgrep binary 'rg-not-real-xyz' was not found. "
            "Expected it installed and on PATH."
        )

    def test_reports_a_timeout(self) -> None:
        # Act: a slow stand-in binary is killed once the timeout elapses.
        outcome = SubprocessRipgrepSearch("sleep").run(["5"], 0.1)

        # Assert
        assert outcome.exit_code is None
        assert outcome.stdout == ""
        assert outcome.stderr == "ripgrep timed out after 0.1 seconds."
