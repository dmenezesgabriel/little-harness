"""A tool spec declares whether the tool needs human approval to run."""

from __future__ import annotations

from little_harness.domain.tool_spec import ToolInputSchema, ToolSpec
from little_harness.domain.values.text_values import ToolName


def spec(*, requires_approval: bool) -> ToolSpec:
    return ToolSpec(
        ToolName("bash"),
        "Run a shell command.",
        ToolInputSchema("a command"),
        requires_approval=requires_approval,
    )


class TestToolSpec:
    def test_does_not_require_approval_by_default(self) -> None:
        # Act / Assert: safe tools opt out simply by not declaring danger.
        plain = ToolSpec(
            ToolName("calculator"), "Evaluate math", ToolInputSchema("expr")
        )

        assert plain.requires_approval is False

    def test_can_declare_that_it_requires_approval(self) -> None:
        # Act / Assert
        assert spec(requires_approval=True).requires_approval is True
