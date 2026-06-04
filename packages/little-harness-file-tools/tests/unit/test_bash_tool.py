from __future__ import annotations

from little_harness.domain.tool_result import ToolRunRequest
from little_harness.domain.values.text_values import ToolInput, ToolName
from little_harness_file_tools.bash_tool import BashTool
from little_harness_file_tools.shell_command_runner import CommandOutcome

from tests.unit.fakes import FakeGuardrail, FakeShellRunner


def bash_request(command: str) -> ToolRunRequest:
    return ToolRunRequest(ToolName("bash"), ToolInput(command))


def succeeding_runner(stdout: str = "ok\n", stderr: str = "") -> FakeShellRunner:
    return FakeShellRunner(CommandOutcome(0, stdout, stderr, timed_out=False))


class TestBashTool:
    def test_advertises_a_sensitive_spec(self) -> None:
        # Act
        spec = BashTool(succeeding_runner(), FakeGuardrail()).spec

        # Assert
        assert spec.name == ToolName("bash")
        assert spec.requires_approval is True

    def test_blocks_a_command_the_guardrail_rejects(self) -> None:
        # Arrange
        runner = succeeding_runner()
        guardrail = FakeGuardrail(reason="blocked: rm -rf /")
        tool = BashTool(runner, guardrail)

        # Act
        result = tool.run(bash_request("rm -rf /"))

        # Assert: the exact command is inspected and never reaches the shell.
        assert result.tool_name == ToolName("bash")
        assert result.succeeded is False
        assert result.output.value == "blocked: rm -rf /"
        assert guardrail.commands == ["rm -rf /"]
        assert runner.commands == []

    def test_runs_an_allowed_command_and_formats_a_success(self) -> None:
        # Arrange
        runner = succeeding_runner(stdout="hello\n")
        tool = BashTool(runner, FakeGuardrail(), timeout_seconds=12.0)

        # Act
        result = tool.run(bash_request("echo hello"))

        # Assert: an empty stderr is omitted, not rendered as a blank section.
        assert result.tool_name == ToolName("bash")
        assert result.succeeded is True
        assert result.output.value == "exit code: 0\nstdout:\nhello\n"
        assert runner.commands == ["echo hello"]
        assert runner.timeouts == [12.0]

    def test_reports_a_non_zero_exit_as_a_failure_with_stderr(self) -> None:
        # Arrange
        runner = FakeShellRunner(CommandOutcome(2, "", "boom\n", timed_out=False))
        tool = BashTool(runner, FakeGuardrail())

        # Act
        result = tool.run(bash_request("false"))

        # Assert: an empty stdout is omitted, not rendered as a blank section.
        assert result.tool_name == ToolName("bash")
        assert result.succeeded is False
        assert result.output.value == "exit code: 2\nstderr:\nboom\n"

    def test_reports_a_timeout_as_a_failure(self) -> None:
        # Arrange
        runner = FakeShellRunner(
            CommandOutcome(None, "", "Command timed out after 0.1 seconds.", True)
        )
        tool = BashTool(runner, FakeGuardrail())

        # Act
        result = tool.run(bash_request("sleep 9"))

        # Assert
        assert result.tool_name == ToolName("bash")
        assert result.succeeded is False
        assert result.output.value == "Command timed out after 0.1 seconds."
