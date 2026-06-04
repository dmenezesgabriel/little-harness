"""Thin boundary around `subprocess` for running a shell command line.

Isolating the only `subprocess` call here keeps the bash tool's formatting and
guardrail logic testable with a fake runner, and confines the security-sensitive
shell execution to a single, auditable place.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class CommandOutcome:
    """The result of running a command: its streams, exit code, and timeout flag.

    `exit_code` is None when the command was killed for exceeding its timeout.

    Example:
        outcome = CommandOutcome(0, "hi\n", "", timed_out=False)
    """

    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool


class ShellCommandRunner(Protocol):
    def run(self, command: str, timeout_seconds: float, /) -> CommandOutcome:
        """Run a command line and capture its outcome.

        Example:
            outcome = runner.run("echo hi", 30.0)
        """
        ...


class SubprocessShellRunner:
    """Runs a command through the system shell, capturing output with a timeout.

    Example:
        SubprocessShellRunner().run("echo hi", 30.0)
    """

    def run(self, command: str, timeout_seconds: float) -> CommandOutcome:
        try:
            # check defaults to False: a non-zero exit is reported, not raised.
            completed = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return CommandOutcome(
                exit_code=None,
                stdout="",
                stderr=f"Command timed out after {timeout_seconds} seconds.",
                timed_out=True,
            )

        return CommandOutcome(
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            timed_out=False,
        )
