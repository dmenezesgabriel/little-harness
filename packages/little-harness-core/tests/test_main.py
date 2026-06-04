"""Tests for the `little-harness` console-script entry point."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from little_harness import __main__
from little_harness.domain.errors import UnknownProviderError


class TestRun:
    def test_prints_the_answer_and_returns_zero(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Arrange: record the argv run forwards to run_cli.
        received: list[list[str]] = []

        def answer(argv: Sequence[str]) -> str:
            received.append(list(argv))
            return "RENDERED ANSWER"

        monkeypatch.setattr(__main__, "run_cli", answer)

        # Act
        status = __main__.run(["--provider", "x"])

        # Assert
        captured = capsys.readouterr()
        assert status == 0
        assert captured.out.strip() == "RENDERED ANSWER"
        assert captured.err == ""
        assert received == [["--provider", "x"]]

    def test_reports_a_concise_error_to_stderr_and_returns_one(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Arrange
        def fail(_argv: Sequence[str]) -> str:
            raise UnknownProviderError("Unknown provider: 'x'.")

        monkeypatch.setattr(__main__, "run_cli", fail)

        # Act
        status = __main__.run([])

        # Assert: no traceback, just one clean line on stderr.
        captured = capsys.readouterr()
        assert status == 1
        assert captured.out == ""
        assert captured.err == "error: Unknown provider: 'x'.\n"

    def test_reraises_the_full_error_under_the_traceback_flag(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        def fail(_argv: Sequence[str]) -> str:
            raise RuntimeError("kaboom")

        monkeypatch.setattr(__main__, "run_cli", fail)

        # Act / Assert: --log surfaces the underlying exception for debugging.
        with pytest.raises(RuntimeError, match="kaboom"):
            __main__.run(["--log"])


class TestMain:
    def test_forwards_cli_args_and_exits_with_the_run_status_code(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange: record the argv main hands to run (the program name is dropped).
        received: list[list[str]] = []

        def recording_run(argv: Sequence[str]) -> int:
            received.append(list(argv))
            return 1

        monkeypatch.setattr(__main__, "run", recording_run)
        monkeypatch.setattr("sys.argv", ["little-harness", "--provider", "x"])

        # Act / Assert
        with pytest.raises(SystemExit) as exc:
            __main__.main()
        assert exc.value.code == 1
        assert received == [["--provider", "x"]]
