from __future__ import annotations

from little_harness.domain.decision import FinalAnswer
from little_harness.domain.tool_result import ToolRunResult
from little_harness.domain.tool_spec import ToolInputSchema, ToolSpec
from little_harness.domain.values.role import USER
from little_harness.domain.values.text_values import (
    MessageContent,
    Prompt,
    ToolName,
    ToolOutput,
)
from little_harness.infrastructure.policy.json_agent_policy import JsonAgentPolicy


class TestJsonAgentPolicy:
    def test_system_prompt_renders_available_tools(self) -> None:
        # Arrange
        spec = ToolSpec(
            ToolName("calculator"),
            "Evaluate math.",
            ToolInputSchema("A numeric expression"),
        )

        # Act
        prompt = JsonAgentPolicy().system_prompt([spec])

        # Assert
        assert "calculator: Evaluate math." in prompt.value

    def test_parse_model_output_delegates_to_the_parser(self) -> None:
        # Arrange
        output = MessageContent(
            '{"action":"final","tool_name":null,"tool_input":null,"answer":"done"}'
        )

        # Act / Assert
        assert JsonAgentPolicy().parse_model_output(output) == FinalAnswer(
            MessageContent("done")
        )

    def test_observation_and_repair_messages_use_the_user_role(self) -> None:
        # Arrange
        policy = JsonAgentPolicy()
        result = ToolRunResult(ToolName("calculator"), ToolOutput("4"), succeeded=True)

        # Act
        observation = policy.build_tool_observation_message(Prompt("q"), result)
        repair = policy.build_repair_message(Prompt("q"), ValueError("bad"))

        # Assert
        assert observation.role == USER
        assert repair.role == USER
        assert "q" in observation.content.value
        # Pin the rendered bodies so dropping the renderer or its error is caught.
        assert "4" in observation.content.value
        assert "bad" in repair.content.value
