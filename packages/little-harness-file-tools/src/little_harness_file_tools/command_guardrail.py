"""Guardrails that block destructive shell commands before they run.

This is the always-on safety net for the bash tool, independent of the human
approval prompt: even an approved session must not run catastrophic commands.
The denylist is injected, so callers can extend or replace it.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol

# Each pattern matches a class of irreversible or system-damaging command. Kept
# deliberately narrow to avoid false positives on ordinary development commands.
DEFAULT_DANGEROUS_PATTERNS: tuple[str, ...] = (
    r"\brm\b(?=.*\s-\S*r)(?=.*\s-\S*f)",  # rm with both recursive and force flags
    r":\s*\(\s*\)\s*\{\s*:\s*\|\s*:",  # fork bomb :(){ :|:& };:
    r"\bmkfs(\.\w+)?\b",  # reformat a filesystem
    r"\bdd\b[^|;&\n]*\bof=/dev/",  # raw write to a block device
    r">\s*/dev/(sd|nvme|hd)",  # clobber a disk device node
    r"\b(shutdown|reboot|halt|poweroff)\b",  # take the host down
    r"\bchmod\s+-\S*R\S*\s+0*777\s+/",  # recursively world-write the root tree
)


class CommandGuardrail(Protocol):
    def rejection_reason(self, command: str, /) -> str | None:
        """Return why a command is blocked, or None if it is allowed.

        Example:
            reason = guardrail.rejection_reason("rm -rf /")
        """
        ...


class DangerousCommandGuardrail:
    """Rejects commands matching any pattern in a denylist of dangerous shapes.

    Example:
        DangerousCommandGuardrail().rejection_reason("rm -rf /")  # a reason
    """

    def __init__(self, patterns: Sequence[str] = DEFAULT_DANGEROUS_PATTERNS) -> None:
        self._patterns = tuple(
            re.compile(pattern, re.IGNORECASE) for pattern in patterns
        )

    def rejection_reason(self, command: str) -> str | None:
        for pattern in self._patterns:
            if pattern.search(command):
                return (
                    f"Command blocked by guardrail (matched {pattern.pattern!r}): "
                    f"{command!r}. Expected a command that is not destructive."
                )

        return None
