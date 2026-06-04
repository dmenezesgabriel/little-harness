from __future__ import annotations

import pytest
from little_harness_file_tools.command_guardrail import DangerousCommandGuardrail

DANGEROUS_COMMANDS = [
    "rm -rf /",
    "rm -fr build",
    "rm -r -f node_modules",
    "rm --recursive --force /tmp/x",
    ":(){ :|:& };:",
    "mkfs.ext4 /dev/sda1",
    "dd if=/dev/zero of=/dev/sda",
    "echo boom > /dev/sda",
    "shutdown now",
    "reboot",
    "chmod -R 777 /",
]

SAFE_COMMANDS = [
    "ls -la",
    "rm file.txt",
    "rm -i note.txt",
    "rm -r build_only",
    "grep -rn TODO src",
    "echo hello",
    "cat README.md",
]


class TestDangerousCommandGuardrail:
    @pytest.mark.parametrize("command", DANGEROUS_COMMANDS)
    def test_rejects_each_dangerous_command(self, command: str) -> None:
        # Act
        reason = DangerousCommandGuardrail().rejection_reason(command)

        # Assert: the reason names the offending command for the operator/model.
        assert reason is not None
        assert "blocked by guardrail" in reason
        assert command in reason

    @pytest.mark.parametrize("command", SAFE_COMMANDS)
    def test_allows_each_ordinary_command(self, command: str) -> None:
        # Act / Assert
        assert DangerousCommandGuardrail().rejection_reason(command) is None

    def test_matches_case_insensitively(self) -> None:
        # Act / Assert: uppercasing must not bypass the denylist.
        assert DangerousCommandGuardrail().rejection_reason("SHUTDOWN NOW") is not None

    def test_uses_an_injected_denylist(self) -> None:
        # Arrange: the denylist is configurable, so callers can extend or replace it.
        guardrail = DangerousCommandGuardrail(patterns=(r"\bsecret\b",))

        # Act / Assert
        assert guardrail.rejection_reason("print secret") is not None
        assert guardrail.rejection_reason("rm -rf /") is None
