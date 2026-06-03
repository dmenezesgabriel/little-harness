from __future__ import annotations

from local_llm.domain.decision import (
    AgentDecision,
    DecisionVisitor,
    FinalAnswer,
    ToolCall,
)
from local_llm.domain.values.text_values import MessageContent, ToolInput, ToolName


class RecordingDecisionVisitor:
    """Visitor that records which branch each decision dispatched to."""

    def __init__(self) -> None:
        self.tool_calls: list[ToolCall] = []
        self.final_answers: list[FinalAnswer] = []

    def visit_tool_call(self, decision: ToolCall) -> str:
        self.tool_calls.append(decision)
        return "tool"

    def visit_final_answer(self, decision: FinalAnswer) -> str:
        self.final_answers.append(decision)
        return "final"


class TestDecisionDispatch:
    def test_tool_call_routes_to_visit_tool_call(self) -> None:
        # Arrange
        visitor = RecordingDecisionVisitor()
        decision: AgentDecision = ToolCall(ToolName("calculator"), ToolInput("2 + 2"))

        # Act
        outcome = decision.accept(visitor)

        # Assert
        assert outcome == "tool"
        assert visitor.tool_calls == [decision]
        assert visitor.final_answers == []

    def test_final_answer_routes_to_visit_final_answer(self) -> None:
        # Arrange
        visitor: DecisionVisitor[str] = RecordingDecisionVisitor()
        decision: AgentDecision = FinalAnswer(MessageContent("done"))

        # Act
        outcome = decision.accept(visitor)

        # Assert
        assert outcome == "final"


class TestDecisionActionName:
    def test_tool_call_names_its_tool(self) -> None:
        # Act / Assert
        assert ToolCall(ToolName("calculator"), ToolInput("2 + 2")).action_name() == (
            "calculator"
        )

    def test_final_answer_names_itself_final(self) -> None:
        # Act / Assert
        assert FinalAnswer(MessageContent("done")).action_name() == "final"
