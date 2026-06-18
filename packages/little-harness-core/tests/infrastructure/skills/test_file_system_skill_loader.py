"""Tests for FileSystemSkillLoader."""

from __future__ import annotations

from pathlib import Path

import pytest
from little_harness.domain.values.skill_values import SkillDescription, SkillName
from little_harness.infrastructure.skills.file_system_skill_loader import (
    FileSystemSkillLoader,
)


class TestFileSystemSkillLoader:
    def test_loads_single_skill(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / ".agents" / "skills" / "my-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\n"
            "name: my-skill\n"
            "description: My custom skill\n"
            "---\n\n"
            "# Instructions\n\n"
            "Do something."
        )

        loader = FileSystemSkillLoader([str(tmp_path / ".agents" / "skills")])
        skills = loader.load_skills()

        assert len(skills) == 1
        assert skills[0].name == SkillName("my-skill")
        assert skills[0].description == SkillDescription("My custom skill")
        assert skills[0].content == "# Instructions\n\nDo something."

    def test_loads_multiple_skills(self, tmp_path: Path) -> None:
        for name in ("skill-a", "skill-b"):
            d = tmp_path / "skills" / name
            d.mkdir(parents=True)
            (d / "SKILL.md").write_text(
                f"---\nname: {name}\ndescription: {name} desc\n---\n\nContent {name}"
            )

        loader = FileSystemSkillLoader([str(tmp_path / "skills")])
        skills = loader.load_skills()

        assert len(skills) == 2
        names = sorted(s.name.value for s in skills)
        assert names == ["skill-a", "skill-b"]

    def test_skips_directories_without_skill_file(self, tmp_path: Path) -> None:
        d = tmp_path / "skills" / "no-skill"
        d.mkdir(parents=True)
        (d / "README.md").write_text("nothing")

        loader = FileSystemSkillLoader([str(tmp_path / "skills")])
        assert loader.load_skills() == []

    def test_skips_directories_that_are_files(self, tmp_path: Path) -> None:
        (tmp_path / "not-a-dir").write_text("file")
        loader = FileSystemSkillLoader([str(tmp_path)])
        assert loader.load_skills() == []

    def test_skips_missing_directory(self) -> None:
        loader = FileSystemSkillLoader(["/nonexistent/path"])
        assert loader.load_skills() == []

    def test_falls_back_to_directory_name(self, tmp_path: Path) -> None:
        d = tmp_path / "skills" / "my-fallback"
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text(
            "---\ndescription: No name in frontmatter\n---\n\nBody"
        )

        loader = FileSystemSkillLoader([str(tmp_path / "skills")])
        skills = loader.load_skills()

        assert len(skills) == 1
        assert skills[0].name.value == "my-fallback"
        assert skills[0].description.value == "No name in frontmatter"

    def test_skips_skill_without_description(self, tmp_path: Path) -> None:
        d = tmp_path / "skills" / "no-desc"
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text("---\nname: no-desc\n---\n\nBody")

        loader = FileSystemSkillLoader([str(tmp_path / "skills")])
        skills = loader.load_skills()

        assert len(skills) == 0

    def test_handles_multiple_skill_directories(self, tmp_path: Path) -> None:
        d1 = tmp_path / "skills-a" / "s1"
        d1.mkdir(parents=True)
        (d1 / "SKILL.md").write_text("---\nname: s1\ndescription: First\n---\n\nOne")

        d2 = tmp_path / "skills-b" / "s2"
        d2.mkdir(parents=True)
        (d2 / "SKILL.md").write_text("---\nname: s2\ndescription: Second\n---\n\nTwo")

        loader = FileSystemSkillLoader(
            [str(tmp_path / "skills-a"), str(tmp_path / "skills-b")]
        )
        skills = loader.load_skills()

        assert len(skills) == 2
        assert skills[0].name.value == "s1"
        assert skills[1].name.value == "s2"

    def test_ignores_broken_file(self, tmp_path: Path) -> None:
        d = tmp_path / "skills" / "broken"
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text("not valid frontmatter")

        loader = FileSystemSkillLoader([str(tmp_path / "skills")])
        skills = loader.load_skills()

        assert len(skills) == 0


class TestFileSystemSkillLoaderWarnings:
    """Diagnostic warnings are logged on parse failures (not silently swallowed)."""

    def test_warns_on_os_error(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        d = tmp_path / "skills" / "unreadable"
        d.mkdir(parents=True)
        skill_file = d / "SKILL.md"
        skill_file.write_text("---\nname: test\ndescription: test\n---\n\nBody")
        skill_file.chmod(0o000)

        loader = FileSystemSkillLoader([str(tmp_path / "skills")])
        loader.load_skills()

        assert len(caplog.records) >= 1
        assert "Failed to read skill file" in caplog.records[0].message

    def test_warns_on_missing_frontmatter(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        d = tmp_path / "skills" / "no-fm"
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text("Just content without frontmatter.")

        loader = FileSystemSkillLoader([str(tmp_path / "skills")])
        loader.load_skills()

        assert len(caplog.records) >= 1
        assert "No valid frontmatter" in caplog.records[0].message

    def test_warns_on_invalid_metadata(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        d = tmp_path / "skills" / "bad-meta"
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text("---\nname: \ndescription: \n---\n\nBody")

        loader = FileSystemSkillLoader([str(tmp_path / "skills")])
        loader.load_skills()

        assert len(caplog.records) >= 1
        assert "Invalid skill metadata" in caplog.records[0].message
