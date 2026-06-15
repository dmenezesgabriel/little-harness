"""Integration tests: skills are loaded and injected into the system prompt."""

from __future__ import annotations

from little_harness.domain.skill import Skill
from little_harness.domain.values.skill_values import SkillDescription, SkillName
from little_harness.domain.values.text_values import Prompt

from tests.application.fakes import (
    DecisionQueuePolicy,
    RecordingAgentTool,
    RecordingChatModel,
    RecordingSkillLoader,
    final_decision,
)
from tests.application.test_agent_runtime import create_runtime


class TestSkillsInSystemPrompt:
    def test_empty_skills_do_not_change_prompt(self) -> None:
        chat_model = RecordingChatModel(["final"])
        policy = DecisionQueuePolicy([final_decision("done")])
        skill_loader = RecordingSkillLoader()
        runtime = create_runtime(
            chat_model, [RecordingAgentTool()], policy, skill_loader=skill_loader
        )

        msg = runtime.build_system_message()

        assert "<available_skills>" not in msg.content.value
        assert skill_loader.load_call_count == 1

    def test_skills_are_appended_to_system_prompt(self) -> None:
        chat_model = RecordingChatModel(["final"])
        policy = DecisionQueuePolicy([final_decision("done")])
        skill = Skill(
            name=SkillName("my-skill"),
            description=SkillDescription("My custom skill."),
            content="# Instructions\nDo the thing.",
            file_path="/tmp/.agents/skills/my-skill/SKILL.md",
        )
        skill_loader = RecordingSkillLoader([skill])
        runtime = create_runtime(
            chat_model, [RecordingAgentTool()], policy, skill_loader=skill_loader
        )

        msg = runtime.build_system_message()

        assert "<available_skills>" in msg.content.value
        assert "<name>my-skill</name>" in msg.content.value
        assert "<description>My custom skill.</description>" in msg.content.value
        assert "/tmp/.agents/skills/my-skill/SKILL.md" in msg.content.value

    def test_multiple_skills_all_listed(self) -> None:
        chat_model = RecordingChatModel(["final"])
        policy = DecisionQueuePolicy([final_decision("done")])
        skills = [
            Skill(
                name=SkillName("skill-a"),
                description=SkillDescription("First."),
                content="A",
                file_path="/a/SKILL.md",
            ),
            Skill(
                name=SkillName("skill-b"),
                description=SkillDescription("Second."),
                content="B",
                file_path="/b/SKILL.md",
            ),
        ]
        skill_loader = RecordingSkillLoader(skills)
        runtime = create_runtime(
            chat_model, [RecordingAgentTool()], policy, skill_loader=skill_loader
        )

        msg = runtime.build_system_message()

        assert msg.content.value.count("<skill>") == 2
        assert "skill-a" in msg.content.value
        assert "skill-b" in msg.content.value

    def test_skills_included_in_run(self) -> None:
        chat_model = RecordingChatModel(["final"])
        policy = DecisionQueuePolicy([final_decision("done")])
        skill = Skill(
            name=SkillName("my-skill"),
            description=SkillDescription("My skill."),
            content="Do something.",
            file_path="/p/SKILL.md",
        )
        skill_loader = RecordingSkillLoader([skill])
        runtime = create_runtime(
            chat_model, [RecordingAgentTool()], policy, skill_loader=skill_loader
        )

        result = runtime.run(Prompt("test"))

        assert result.answer.value == "done"
        # Skill loader was consulted during build_system_message
        assert skill_loader.load_call_count == 1
