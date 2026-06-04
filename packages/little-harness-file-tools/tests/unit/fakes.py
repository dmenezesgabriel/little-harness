"""Named test doubles for the bash tool's injected collaborators."""

from __future__ import annotations

from little_harness_file_tools.shell_command_runner import CommandOutcome


class FakeShellRunner:
    """Returns a preset outcome and records the command and timeout it received."""

    def __init__(self, outcome: CommandOutcome) -> None:
        self._outcome = outcome
        self.commands: list[str] = []
        self.timeouts: list[float] = []

    def run(self, command: str, timeout_seconds: float) -> CommandOutcome:
        self.commands.append(command)
        self.timeouts.append(timeout_seconds)
        return self._outcome


class FakeGuardrail:
    """Returns a preset rejection reason and records the commands it inspected."""

    def __init__(self, reason: str | None = None) -> None:
        self._reason = reason
        self.commands: list[str] = []

    def rejection_reason(self, command: str) -> str | None:
        self.commands.append(command)
        return self._reason
