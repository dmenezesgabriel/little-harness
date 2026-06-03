from __future__ import annotations

import pytest

from local_llm.domain.decision import FinalAnswer
from local_llm.domain.step import AgentStep
from local_llm.domain.steps import AgentSteps
from local_llm.domain.tool_spec import ToolExamples
from local_llm.domain.values.numeric_values import Iteration
from local_llm.domain.values.text_values import MessageContent


def sample_step() -> AgentStep:
    return AgentStep(
        Iteration(1),
        MessageContent("output"),
        FinalAnswer(MessageContent("done")),
        MessageContent("observation"),
    )


class TestAgentSteps:
    def test_starts_empty_and_grows_immutably(self) -> None:
        # Arrange
        empty = AgentSteps()

        # Act
        grown = empty.with_step(sample_step())

        # Assert
        assert empty.is_empty() is True
        assert grown.is_empty() is False
        assert list(grown) == [sample_step()]


class TestToolExamples:
    def test_reports_emptiness_and_first(self) -> None:
        # Arrange
        examples = ToolExamples(("144 / 12", "2 ** 8"))

        # Act / Assert
        assert examples.is_empty() is False
        assert examples.first() == "144 / 12"
        assert examples.joined(", ") == "144 / 12, 2 ** 8"

    def test_first_rejects_empty_collection(self) -> None:
        # Arrange
        examples = ToolExamples()

        # Act / Assert
        assert examples.is_empty() is True
        assert len(examples) == 0
        with pytest.raises(ValueError, match="ToolExamples is empty"):
            examples.first()
