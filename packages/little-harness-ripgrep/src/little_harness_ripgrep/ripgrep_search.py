"""Thin boundary around the ripgrep (`rg`) binary, run without a shell.

Isolating the only `subprocess` call here keeps the tool's argument parsing and
exit-code interpretation testable with a fake, and confines process execution to
one auditable place. The binary name is injectable so a missing-binary path can
be exercised in tests and an alternate `rg` can be configured.
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

RIPGREP_BINARY = "rg"


@dataclass(frozen=True)
class RipgrepOutcome:
    """ripgrep's result. `exit_code` is None when `rg` is absent or timed out.

    By ripgrep's convention: 0 means matches, 1 means no matches, 2+ an error.

    Example:
        outcome = RipgrepOutcome(0, "app.py:1:TODO\n", "")
    """

    exit_code: int | None
    stdout: str
    stderr: str


class RipgrepSearch(Protocol):
    def run(
        self, arguments: Sequence[str], timeout_seconds: float, /
    ) -> RipgrepOutcome:
        """Run `rg` with the given arguments and capture its outcome.

        Example:
            outcome = search.run(["TODO", "src"], 30.0)
        """
        ...


class SubprocessRipgrepSearch:
    """Runs the real `rg` binary, capturing output with a timeout.

    Example:
        SubprocessRipgrepSearch().run(["TODO", "src"], 30.0)
    """

    def __init__(self, binary: str = RIPGREP_BINARY) -> None:
        self._binary = binary

    def run(self, arguments: Sequence[str], timeout_seconds: float) -> RipgrepOutcome:
        try:
            completed = subprocess.run(
                [self._binary, *arguments],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except FileNotFoundError:
            return RipgrepOutcome(
                None,
                "",
                f"ripgrep binary {self._binary!r} was not found. "
                "Expected it installed and on PATH.",
            )
        except subprocess.TimeoutExpired:
            return RipgrepOutcome(
                None, "", f"ripgrep timed out after {timeout_seconds} seconds."
            )

        return RipgrepOutcome(completed.returncode, completed.stdout, completed.stderr)
