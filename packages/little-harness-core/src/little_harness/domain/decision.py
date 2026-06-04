"""Polymorphic agent decisions, dispatched through a visitor.

Replacing a `kind` discriminator with a type hierarchy lets the runtime advance
the loop without inspecting tags: each decision routes itself to the matching
visitor method.

Example:
    outcome = decision.accept(loop_visitor)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol

from little_harness.domain.values.text_values import MessageContent, ToolInput, ToolName


class DecisionVisitor[T](Protocol):
    def visit_tool_call(self, decision: ToolCall) -> T:
        """Handle a request to call a tool."""
        ...

    def visit_final_answer(self, decision: FinalAnswer) -> T:
        """Handle a final answer that ends the loop."""
        ...


class AgentDecision(ABC):
    @abstractmethod
    def accept[T](self, visitor: DecisionVisitor[T]) -> T:
        """Dispatch to the visitor method for this decision's concrete type."""
        ...

    @abstractmethod
    def action_name(self) -> str:
        """Name this decision's action (the tool's name, or "final")."""
        ...


@dataclass(frozen=True)
class ToolCall(AgentDecision):
    tool_name: ToolName
    tool_input: ToolInput

    def accept[T](self, visitor: DecisionVisitor[T]) -> T:
        return visitor.visit_tool_call(self)

    def action_name(self) -> str:
        return self.tool_name.value


@dataclass(frozen=True)
class FinalAnswer(AgentDecision):
    answer: MessageContent

    def accept[T](self, visitor: DecisionVisitor[T]) -> T:
        return visitor.visit_final_answer(self)

    def action_name(self) -> str:
        return "final"
