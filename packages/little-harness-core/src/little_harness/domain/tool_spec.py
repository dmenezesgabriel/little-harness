"""Describes a tool exposed to the agent: name, purpose, and input schema."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field

from little_harness.domain.values.text_values import ToolName


@dataclass(frozen=True)
class ToolExamples:
    """First-class collection of example tool inputs used in prompts.

    Example:
        examples = ToolExamples(("144 / 12", "2 ** 8"))

    """

    _values: tuple[str, ...] = ()

    def is_empty(self) -> bool:
        """Return `True` when no examples have been registered."""
        return len(self._values) == 0

    def first(self) -> str:
        """Return the first example, or raise on an empty collection."""
        if self.is_empty():
            raise ValueError("ToolExamples is empty. Expected at least one example.")

        return self._values[0]

    def joined(self, separator: str) -> str:
        """Concatenate all examples with `separator`."""
        return separator.join(self._values)

    def __iter__(self) -> Iterator[str]:
        """Yield each example string in order."""
        return iter(self._values)

    def __len__(self) -> int:
        """Return the number of examples."""
        return len(self._values)


@dataclass(frozen=True)
class ToolInputSchema:
    """Human-readable description of a tool's input plus optional examples.

    Example:
        schema = ToolInputSchema("A numeric expression", ToolExamples(("2 + 2",)))

    """

    description: str
    examples: ToolExamples = field(default_factory=ToolExamples)
    json_schema: Mapping[str, object] | None = None


@dataclass(frozen=True)
class ToolSpec:
    """The advertised contract of a tool.

    `requires_approval` lets a tool declare that it performs sensitive actions
    (writing files, running shell commands) so the runtime asks a human before
    each call. The core never names tools; danger is the tool's own declaration.

    Example:
        spec = ToolSpec(ToolName("calculator"), "Evaluate math", schema)

    """

    name: ToolName
    description: str
    input_schema: ToolInputSchema
    requires_approval: bool = False
