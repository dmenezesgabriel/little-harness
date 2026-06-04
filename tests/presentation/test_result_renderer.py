from __future__ import annotations

from local_llm.domain.decision import AgentDecision, FinalAnswer, ToolCall
from local_llm.domain.result import AgentResult
from local_llm.domain.step import AgentStep
from local_llm.domain.steps import AgentSteps
from local_llm.domain.values.numeric_values import ElapsedSeconds, Iteration
from local_llm.domain.values.text_values import (
    MessageContent,
    ToolInput,
    ToolName,
)
from local_llm.presentation.cli.result_renderer import (
    ResultRenderer,
    format_step_action,
)


def step_with(decision: AgentDecision | None) -> AgentStep:
    return AgentStep(
        Iteration(1),
        MessageContent("output"),
        decision,
        MessageContent("observation"),
    )


class TestFormatStepAction:
    def test_names_a_tool_call_by_its_tool(self) -> None:
        # Arrange
        step = step_with(ToolCall(ToolName("calculator"), ToolInput("2 + 2")))

        # Act / Assert
        assert format_step_action(step) == "calculator"

    def test_names_a_final_answer_final(self) -> None:
        # Arrange
        step = step_with(FinalAnswer(MessageContent("done")))

        # Act / Assert
        assert format_step_action(step) == "final"

    def test_names_a_repaired_step_repair(self) -> None:
        # Arrange
        step = step_with(None)

        # Act / Assert
        assert format_step_action(step) == "repair"


class TestResultRenderer:
    def test_renders_answer_and_elapsed_without_steps(self) -> None:
        # Arrange
        result = AgentResult(
            MessageContent("the answer"), ElapsedSeconds(1.5), AgentSteps()
        )

        # Act
        text = ResultRenderer().render(result)

        # Assert: exact layout — a blank line separates answer from elapsed.
        assert text == "the answer\n\nElapsed: 1.50s"

    def test_renders_steps_when_present(self) -> None:
        # Arrange
        decision = ToolCall(ToolName("calculator"), ToolInput("2 + 2"))
        step = AgentStep(
            Iteration(1), MessageContent("output"), decision, MessageContent("4")
        )
        result = AgentResult(
            MessageContent("done"), ElapsedSeconds(0.5), AgentSteps().with_step(step)
        )

        # Act
        text = ResultRenderer().render(result)

        # Assert: exact layout, including the blank-line separators.
        assert text == (
            "done\n\nElapsed: 0.50s\n\n"
            "Agent steps:\n\nStep 1\nAction: calculator\nObservation: 4"
        )
