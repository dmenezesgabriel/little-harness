"""Ports for loading skills — SkillLoader protocol and supporting types."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from little_harness.domain.skill import Skill


class SkillDiagnosticCode(StrEnum):
    """Stable codes for skill loading diagnostics."""

    FILE_INFO_FAILED = "file_info_failed"
    LIST_FAILED = "list_failed"
    READ_FAILED = "read_failed"
    PARSE_FAILED = "parse_failed"
    INVALID_METADATA = "invalid_metadata"


@dataclass(frozen=True)
class SkillDiagnostic:
    """Warning produced while loading skills.

    Example:
        diagnostic = SkillDiagnostic(
            code=SkillDiagnosticCode.READ_FAILED,
            message="Permission denied",
            file_path="/home/user/.agents/skills/broken/SKILL.md",
        )
    """

    code: SkillDiagnosticCode
    message: str
    file_path: str


class SkillLoader(Protocol):
    """Load skills from a configured set of directories.

    Example:
        loader = FileSystemSkillLoader(["/home/user/.agents/skills"])
        skills = loader.load_skills()
    """

    def load_skills(self) -> Sequence[Skill]:
        """Load all discoverable skills.

        Returns an empty sequence when no skills are found or the directory
        does not exist.
        """
        ...
