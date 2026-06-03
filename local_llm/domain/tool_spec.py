"""Describes a tool exposed to the agent: name, purpose, and input schema."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from local_llm.domain.values.text_values import ToolName


@dataclass(frozen=True)
class ToolExamples:
    """First-class collection of example tool inputs used in prompts.

    Example:
        examples = ToolExamples(("144 / 12", "2 ** 8"))
    """

    _values: tuple[str, ...] = ()

    def is_empty(self) -> bool:
        return len(self._values) == 0

    def first(self) -> str:
        if self.is_empty():
            raise ValueError("ToolExamples is empty. Expected at least one example.")

        return self._values[0]

    def joined(self, separator: str) -> str:
        return separator.join(self._values)

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)


@dataclass(frozen=True)
class ToolInputSchema:
    """Human-readable description of a tool's input plus optional examples.

    Example:
        schema = ToolInputSchema("A numeric expression", ToolExamples(("2 + 2",)))
    """

    description: str
    examples: ToolExamples = field(default_factory=ToolExamples)


@dataclass(frozen=True)
class ToolSpec:
    """The advertised contract of a tool.

    Example:
        spec = ToolSpec(ToolName("calculator"), "Evaluate math", schema)
    """

    name: ToolName
    description: str
    input_schema: ToolInputSchema
