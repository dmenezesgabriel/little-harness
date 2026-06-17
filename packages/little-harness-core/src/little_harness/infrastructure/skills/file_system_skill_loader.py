"""Skill loader that reads skills from SKILL.md files in configured directories."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

from little_harness.domain.skill import Skill
from little_harness.domain.values.skill_values import SkillDescription, SkillName

logger = logging.getLogger(__name__)


class FileSystemSkillLoader:
    """Walk directories, find SKILL.md files, parse frontmatter, and load skills.

    Example:
        loader = FileSystemSkillLoader([".agents/skills"])
        skills = loader.load_skills()

    Missing directories are silently skipped.
    """

    def __init__(self, skill_dirs: Sequence[str]) -> None:
        """See class docstring."""
        self._directories = [Path(d) for d in skill_dirs]

    def load_skills(self) -> Sequence[Skill]:
        """Load all skills from the configured directories."""
        skills: list[Skill] = []

        for directory in self._directories:
            if not directory.is_dir():
                continue

            for skill_dir in sorted(directory.iterdir()):
                if not skill_dir.is_dir():
                    continue

                skill_file = skill_dir / "SKILL.md"
                if not skill_file.is_file():
                    continue

                skill = self._parse_skill_file(skill_file)
                if skill is not None:
                    skills.append(skill)

        return skills

    @staticmethod
    def _parse_skill_file(file_path: Path) -> Skill | None:
        """Parse a SKILL.md file and return a Skill, or None on failure."""
        try:
            content = file_path.read_text(encoding="utf-8")
        except OSError as error:
            logger.warning("Failed to read skill file %s: %s", file_path, error)
            return None

        frontmatter, body = _parse_frontmatter(content)
        if frontmatter is None:
            logger.warning(
                "No valid frontmatter found in skill file %s. "
                "Expected a SKILL.md starting with ---",
                file_path,
            )
            return None

        name_str = frontmatter.get("name") or file_path.parent.name
        desc_str = frontmatter.get("description", "")

        try:
            name = SkillName(name_str)
            description = SkillDescription(desc_str)
        except ValueError as error:
            logger.warning(
                "Invalid skill metadata in %s: %s. "
                "Expected a non-empty name and description.",
                file_path,
                error,
            )
            return None

        return Skill(
            name=name,
            description=description,
            content=body,
            file_path=str(file_path),
        )


def _parse_frontmatter(
    content: str,
) -> tuple[dict[str, str] | None, str]:
    """Parse YAML-like frontmatter between --- delimiters.

    Returns (frontmatter_dict, body) or (None, original_content) if no
    valid frontmatter is found.
    """
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")

    if not normalized.startswith("---"):
        return None, normalized

    end_index = normalized.find("\n---", 3)
    if end_index == -1:
        return None, normalized

    yaml_text = normalized[4:end_index]
    body = normalized[end_index + 4 :].strip()

    frontmatter: dict[str, str] = {}
    for raw_line in yaml_text.split("\n"):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" in stripped:
            key, _, value = stripped.partition(":")
            key = key.strip()
            value = value.strip().strip("\"'")
            if key:
                frontmatter[key] = value

    return frontmatter, body
