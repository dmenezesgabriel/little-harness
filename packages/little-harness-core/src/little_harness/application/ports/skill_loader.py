"""Port for loading skills — SkillLoader protocol."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from little_harness.domain.skill import Skill


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
