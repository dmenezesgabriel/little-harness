"""Renders an `AgentResult` as plain text for the CLI user."""

from __future__ import annotations

from local_llm.domain.result import AgentResult
from local_llm.domain.step import AgentStep
from local_llm.domain.steps import AgentSteps


class ResultRenderer:
    """Produces the user-facing text for a completed run.

    Example:
        text = ResultRenderer().render(result)
    """

    def render(self, result: AgentResult) -> str:
        lines = [result.answer.value, "", f"Elapsed: {result.elapsed.value:.2f}s"]

        if result.steps.is_empty():
            return "\n".join(lines)

        return "\n".join(lines + render_steps(result.steps))


def render_steps(steps: AgentSteps) -> list[str]:
    lines = ["", "Agent steps:"]

    for step in steps:
        lines.append("")
        lines.append(f"Step {step.iteration.value}")
        lines.append(f"Action: {format_step_action(step)}")
        lines.append(f"Observation: {step.observation.value}")

    return lines


def format_step_action(step: AgentStep) -> str:
    if step.decision is None:
        return "repair"

    return step.decision.action_name()
