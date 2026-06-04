"""Text value objects: typed, validated wrappers around domain strings."""

from __future__ import annotations

from dataclasses import dataclass

from little_harness.domain.values.guards import require_non_empty_text


@dataclass(frozen=True)
class Prompt:
    """The user question that drives an agent run. Must be non-empty.

    Example:
        prompt = Prompt("What is 2 + 2?")
    """

    value: str

    def __post_init__(self) -> None:
        require_non_empty_text(self.value, "Prompt")


@dataclass(frozen=True)
class RunId:
    """Correlation id for a single agent run. Non-empty and trimmed.

    Ties together every observability event of one run (logs, metrics, spans).

    Example:
        run_id = RunId("a1b2c3d4")
    """

    value: str

    def __post_init__(self) -> None:
        normalized = require_non_empty_text(self.value, "RunId")
        object.__setattr__(self, "value", normalized)


@dataclass(frozen=True)
class MessageContent:
    """Free-form text of a chat message (system/user/assistant or observation).

    Example:
        content = MessageContent("You are a strict JSON agent.")
    """

    value: str


@dataclass(frozen=True)
class ToolName:
    """A tool identifier. Non-empty and trimmed.

    Example:
        name = ToolName("calculator")
    """

    value: str

    def __post_init__(self) -> None:
        normalized = require_non_empty_text(self.value, "ToolName")
        object.__setattr__(self, "value", normalized)


@dataclass(frozen=True)
class ToolInput:
    """Raw model-provided input for a tool. Trimmed; may be empty.

    Example:
        tool_input = ToolInput("144 / 12")
    """

    value: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", self.value.strip())


@dataclass(frozen=True)
class ToolOutput:
    """The text a tool produced for an observation.

    Example:
        output = ToolOutput("12")
    """

    value: str
