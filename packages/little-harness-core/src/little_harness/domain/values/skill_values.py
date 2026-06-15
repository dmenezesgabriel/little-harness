"""Skill value objects — validated wrappers around skill identifiers."""

from __future__ import annotations

from dataclasses import dataclass

from little_harness.domain.values.guards import require_non_empty_text


@dataclass(frozen=True)
class SkillName:
    """A skill identifier. Non-empty, trimmed.

    Example:
        name = SkillName("hf-cli")
    """

    value: str

    def __post_init__(self) -> None:
        """Validate and normalize the skill name."""
        normalized = require_non_empty_text(self.value, "SkillName")
        object.__setattr__(self, "value", normalized)


@dataclass(frozen=True)
class SkillDescription:
    """A skill's description. Non-empty, trimmed.

    Example:
        desc = SkillDescription("Hugging Face CLI for managing repos.")
    """

    value: str

    def __post_init__(self) -> None:
        """Validate and normalize the skill description."""
        normalized = require_non_empty_text(self.value, "SkillDescription")
        object.__setattr__(self, "value", normalized)
