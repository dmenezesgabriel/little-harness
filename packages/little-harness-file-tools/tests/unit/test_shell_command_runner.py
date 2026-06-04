"""Exercises the real subprocess boundary with fast, hermetic shell builtins."""

from __future__ import annotations

from little_harness_file_tools.shell_command_runner import SubprocessShellRunner

FAILING_EXIT_CODE = 3


class TestSubprocessShellRunner:
    def test_captures_stdout_and_a_zero_exit_code(self) -> None:
        # Act: shell semantics and text capture are exercised end to end.
        outcome = SubprocessShellRunner().run("echo hello", 5.0)

        # Assert
        assert outcome.exit_code == 0
        assert outcome.stdout == "hello\n"
        assert outcome.stderr == ""
        assert outcome.timed_out is False

    def test_reports_a_non_zero_exit_code(self) -> None:
        # Act: the runner must surface failures, not raise on them.
        outcome = SubprocessShellRunner().run(f"exit {FAILING_EXIT_CODE}", 5.0)

        # Assert
        assert outcome.exit_code == FAILING_EXIT_CODE
        assert outcome.timed_out is False

    def test_marks_a_command_that_exceeds_its_timeout(self) -> None:
        # Act: a slow command is killed and reported as timed out.
        outcome = SubprocessShellRunner().run("sleep 5", 0.1)

        # Assert
        assert outcome.timed_out is True
        assert outcome.exit_code is None
        assert outcome.stdout == ""
        assert "timed out" in outcome.stderr
