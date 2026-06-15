"""Tests for Skill entity."""

from __future__ import annotations

import pytest
from little_harness.domain.skill import Skill
from little_harness.domain.values.skill_values import SkillDescription, SkillName


class TestSkill:
    def test_holds_all_fields(self) -> None:
        skill = Skill(
            name=SkillName("my-skill"),
            description=SkillDescription("Does something."),
            content="Some instructions here.",
            file_path="/path/to/SKILL.md",
        )
        assert skill.name.value == "my-skill"
        assert skill.description.value == "Does something."
        assert skill.content == "Some instructions here."
        assert skill.file_path == "/path/to/SKILL.md"

    def test_equality(self) -> None:
        skill1 = Skill(
            name=SkillName("a"),
            description=SkillDescription("desc"),
            content="body",
            file_path="/a/SKILL.md",
        )
        skill2 = Skill(
            name=SkillName("a"),
            description=SkillDescription("desc"),
            content="body",
            file_path="/a/SKILL.md",
        )
        assert skill1 == skill2

    def test_immutable(self) -> None:
        skill = Skill(
            name=SkillName("s"),
            description=SkillDescription("d"),
            content="c",
            file_path="f",
        )
        with pytest.raises(AttributeError):
            skill.name = SkillName("other")  # type: ignore[misc]
