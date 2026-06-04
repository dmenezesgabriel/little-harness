"""Each hook decision routes itself to the matching visitor method."""

from __future__ import annotations

from local_llm.domain.hook_decision import (
    Block,
    HookDecision,
    InjectContext,
    Proceed,
)
from local_llm.domain.values.text_values import MessageContent


class LabelingVisitor:
    """Visitor that names the variant it was dispatched to."""

    def visit_proceed(self, _decision: Proceed) -> str:
        return "proceed"

    def visit_inject_context(self, decision: InjectContext) -> str:
        return f"inject:{decision.content.value}"

    def visit_block(self, decision: Block) -> str:
        return f"block:{decision.reason.value}"


class TestHookDecisionDispatch:
    def test_proceed_dispatches_to_visit_proceed(self) -> None:
        decision: HookDecision = Proceed()

        assert decision.accept(LabelingVisitor()) == "proceed"

    def test_inject_context_dispatches_with_its_content(self) -> None:
        decision: HookDecision = InjectContext(MessageContent("note"))

        assert decision.accept(LabelingVisitor()) == "inject:note"

    def test_block_dispatches_with_its_reason(self) -> None:
        decision: HookDecision = Block(MessageContent("denied"))

        assert decision.accept(LabelingVisitor()) == "block:denied"
