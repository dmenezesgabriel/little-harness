"""Tests for the `little-harness` console-script entry point."""

from __future__ import annotations

import pytest
from little_harness import __main__


class TestMain:
    def test_prints_the_rendered_run_output(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Arrange: stub the wired run so the entry point is tested in isolation.
        monkeypatch.setattr(__main__, "run_cli", lambda: "RENDERED ANSWER")

        # Act
        __main__.main()

        # Assert
        assert capsys.readouterr().out.strip() == "RENDERED ANSWER"
