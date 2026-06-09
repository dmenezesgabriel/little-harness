"""Polymorphic lifecycle-hook decisions, dispatched through a visitor.

A hook answers each lifecycle point with one of three decisions, mirroring
`domain/decision.py` so the runtime advances without inspecting tags:

- `Proceed`        — do the action unchanged.
- `InjectContext`  — append context, then do the action.
- `Block`          — do not do the action; the runtime takes the alternative path.

Example:
    outcome = decision.accept(my_visitor)

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol

from little_harness.domain.values.text_values import MessageContent


class HookDecisionVisitor[T](Protocol):
    """Visitor interface for dispatching `HookDecision` subtypes."""

    # Decisions are passed positionally, so visitors need not echo the arg name
    # (lets a no-op branch use `_decision` without breaking protocol conformance).
    def visit_proceed(self, decision: Proceed, /) -> T:
        """Handle a decision to proceed unchanged."""
        ...

    def visit_inject_context(self, decision: InjectContext, /) -> T:
        """Handle a decision to inject context before proceeding."""
        ...

    def visit_block(self, decision: Block, /) -> T:
        """Handle a decision to block the action."""
        ...


class HookDecision(ABC):
    """Base class for polymorphic lifecycle-hook decisions dispatched by `accept`."""

    @abstractmethod
    def accept[T](self, visitor: HookDecisionVisitor[T]) -> T:
        """Dispatch to the visitor method for this decision's concrete type."""
        ...


@dataclass(frozen=True)
class Proceed(HookDecision):
    """A hook decision to proceed with the action unchanged."""

    def accept[T](self, visitor: HookDecisionVisitor[T]) -> T:
        """Dispatch to `visit_proceed` on the given visitor."""
        return visitor.visit_proceed(self)


@dataclass(frozen=True)
class InjectContext(HookDecision):
    """A hook decision to inject context before proceeding."""

    content: MessageContent

    def accept[T](self, visitor: HookDecisionVisitor[T]) -> T:
        """Dispatch to `visit_inject_context` on the given visitor."""
        return visitor.visit_inject_context(self)


@dataclass(frozen=True)
class Block(HookDecision):
    """A hook decision to block the action."""

    reason: MessageContent

    def accept[T](self, visitor: HookDecisionVisitor[T]) -> T:
        """Dispatch to `visit_block` on the given visitor."""
        return visitor.visit_block(self)
