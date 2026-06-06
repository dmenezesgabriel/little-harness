"""Tool that runs a shell command behind a guardrail, with human approval."""

from __future__ import annotations

from little_harness.domain.tool_result import ToolRunRequest, ToolRunResult
from little_harness.domain.tool_spec import ToolExamples, ToolInputSchema, ToolSpec
from little_harness.domain.values.text_values import ToolName, ToolOutput

from little_harness_file_tools.command_guardrail import CommandGuardrail
from little_harness_file_tools.shell_command_runner import (
    CommandOutcome,
    ShellCommandRunner,
)

DEFAULT_TIMEOUT_SECONDS = 30.0


class BashTool:
    """Runs a shell command and reports its output, exit code, and any failure.

    The runner and guardrail are injected; the guardrail blocks destructive
    commands and the spec declares approval is required.

    Example:
        BashTool(runner, guardrail).run(request)  # raw_input = "ls -la"
    """

    def __init__(
        self,
        runner: ShellCommandRunner,
        guardrail: CommandGuardrail,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._runner = runner
        self._guardrail = guardrail
        self._timeout_seconds = timeout_seconds

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            ToolName("bash"),
            "Run a shell command line and return its output and exit code.",
            ToolInputSchema(
                "A shell command line to execute.",
                ToolExamples(("ls -la", "grep -rn TODO src")),
                {"type": "string"},
            ),
            requires_approval=True,
        )

    def run(self, request: ToolRunRequest) -> ToolRunResult:
        command = request.raw_input.value
        reason = self._guardrail.rejection_reason(command)

        if reason is not None:
            return ToolRunResult(request.tool_name, ToolOutput(reason), succeeded=False)

        outcome = self._runner.run(command, self._timeout_seconds)
        return self._to_result(request.tool_name, outcome)

    def _to_result(self, tool_name: ToolName, outcome: CommandOutcome) -> ToolRunResult:
        if outcome.timed_out:
            return ToolRunResult(tool_name, ToolOutput(outcome.stderr), succeeded=False)

        return ToolRunResult(
            tool_name,
            ToolOutput(format_outcome(outcome)),
            succeeded=outcome.exit_code == 0,
        )


def format_outcome(outcome: CommandOutcome) -> str:
    sections = [f"exit code: {outcome.exit_code}"]

    if outcome.stdout != "":
        sections.append(f"stdout:\n{outcome.stdout}")

    if outcome.stderr != "":
        sections.append(f"stderr:\n{outcome.stderr}")

    return "\n".join(sections)
