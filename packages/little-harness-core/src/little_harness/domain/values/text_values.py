"""Text value objects: typed, validated wrappers around domain strings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from little_harness.domain.values.guards import require_non_empty_text

if TYPE_CHECKING:
    from little_harness.domain.values.thinking import ThinkingContent


@dataclass(frozen=True)
class Prompt:
    """The user question that drives an agent run. Must be non-empty.

    Example:
        prompt = Prompt("What is 2 + 2?")

    """

    value: str

    def __post_init__(self) -> None:
        """Validate that prompt is non-empty."""
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
        """Validate and normalize the run id."""
        normalized = require_non_empty_text(self.value, "RunId")
        object.__setattr__(self, "value", normalized)


@dataclass(frozen=True)
class SessionId:
    """Identifier for a conversation session. Non-empty and trimmed.

    Ties together the durable history of an agent across multiple runs.

    Example:
        session_id = SessionId("test-session")

    """

    value: str

    def __post_init__(self) -> None:
        """Validate and normalize the session id."""
        normalized = require_non_empty_text(self.value, "SessionId")
        object.__setattr__(self, "value", normalized)


@dataclass(frozen=True)
class MessageContent:
    """Free-form text of a chat message, with optional reasoning content.

    When the model supports thinking, `thinking` holds the chain-of-thought
    tokens the model produced internally before the visible answer.

    Example:
        content = MessageContent("You are a strict JSON agent.")

    """

    value: str
    thinking: ThinkingContent | None = None  # type: ignore[misc]


@dataclass(frozen=True)
class ToolName:
    """A tool identifier. Non-empty and trimmed.

    Example:
        name = ToolName("calculator")

    """

    value: str

    def __post_init__(self) -> None:
        """Validate and normalize the tool name."""
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
        """Strip whitespace from the raw tool input."""
        object.__setattr__(self, "value", self.value.strip())


@dataclass(frozen=True)
class ToolOutput:
    """The text a tool produced for an observation.

    Example:
        output = ToolOutput("12")

    """

    value: str
