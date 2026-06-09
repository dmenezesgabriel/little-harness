"""Renders an `AgentResult` as plain text for the CLI user."""

from __future__ import annotations

from little_harness.domain.result import AgentResult
from little_harness.domain.step import AgentStep
from little_harness.domain.steps import AgentSteps


class ResultRenderer:
    """Produces the user-facing text for a completed run.

    Example:
        text = ResultRenderer().render(result)

    """

    def render(self, result: AgentResult) -> str:
        """Render an ``AgentResult`` as user-facing text."""
        lines = [result.answer.value, "", f"Elapsed: {result.elapsed.value:.2f}s"]

        if result.steps.is_empty():
            return "\n".join(lines)

        return "\n".join(lines + render_steps(result.steps))


def render_steps(steps: AgentSteps) -> list[str]:
    """Build a human-readable list of lines from agent step data."""
    lines = ["", "Agent steps:"]

    for step in steps:
        lines.append("")
        lines.append(f"Step {step.iteration.value}")
        lines.append(f"Action: {format_step_action(step)}")
        lines.append(f"Observation: {step.observation.value}")

    return lines


def format_step_action(step: AgentStep) -> str:
    """Format a step's action as a human-readable string."""
    if step.decision is None:
        return "repair"

    return step.decision.action_name()
