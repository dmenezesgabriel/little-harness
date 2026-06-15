"""Tests for SkillName and SkillDescription value objects."""

from __future__ import annotations

import pytest
from little_harness.domain.values.skill_values import SkillDescription, SkillName


class TestSkillName:
    def test_accepts_valid_name(self) -> None:
        name = SkillName("hf-cli")
        assert name.value == "hf-cli"

    def test_strips_whitespace(self) -> None:
        name = SkillName("  my-skill  ")
        assert name.value == "my-skill"

    def test_rejects_empty_string(self) -> None:
        with pytest.raises(ValueError, match="SkillName is empty"):
            SkillName("")

    def test_rejects_whitespace_only(self) -> None:
        with pytest.raises(ValueError, match="SkillName is empty"):
            SkillName("   ")


class TestSkillDescription:
    def test_accepts_valid_description(self) -> None:
        desc = SkillDescription("Useful for ML tasks.")
        assert desc.value == "Useful for ML tasks."

    def test_strips_whitespace(self) -> None:
        desc = SkillDescription("  Some skill  ")
        assert desc.value == "Some skill"

    def test_rejects_empty_string(self) -> None:
        with pytest.raises(ValueError, match="SkillDescription is empty"):
            SkillDescription("")

    def test_rejects_whitespace_only(self) -> None:
        with pytest.raises(ValueError, match="SkillDescription is empty"):
            SkillDescription("   ")
