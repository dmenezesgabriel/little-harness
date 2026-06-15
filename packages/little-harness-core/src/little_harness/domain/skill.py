"""Skill entity — a loaded skill with metadata and content."""

from __future__ import annotations

from dataclasses import dataclass

from little_harness.domain.values.skill_values import SkillDescription, SkillName


@dataclass(frozen=True)
class Skill:
    r"""A skill loaded from the filesystem with metadata and content.

    Example:
        skill = Skill(
            name=SkillName("hf-cli"),
            description=SkillDescription("Hugging Face CLI."),
            content="# hf-cli\nUse this tool...",
            file_path="/home/user/.agents/skills/hf-cli/SKILL.md",
        )
    """

    name: SkillName
    description: SkillDescription
    content: str
    file_path: str
