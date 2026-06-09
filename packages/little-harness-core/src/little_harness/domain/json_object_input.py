"""Parses a tool's raw string input into a validated JSON object.

The `AgentTool` contract delivers one raw string; tools that need several fields
(for example a path plus content) carry them as a JSON object. This value object
is the one shared place that turns that string into typed, present fields with
actionable errors, so each tool plugin does not reinvent it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast


@dataclass(frozen=True)
class JsonObjectInput:
    """A parsed JSON object with typed field access.

    Example:
        fields = JsonObjectInput.parse('{"path": "a.txt"}')
        path = fields.required_text("path")

    """

    fields: Mapping[str, object]

    @classmethod
    def parse(cls, raw: str) -> JsonObjectInput:
        """Parse a raw JSON string into a `JsonObjectInput`."""
        try:
            loaded: object = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid JSON tool input: {raw!r}. Expected a JSON object."
            ) from error

        if not isinstance(loaded, dict):
            raise ValueError(
                f"Invalid JSON tool input: {raw!r}. "
                f"Expected a JSON object, got {type(loaded).__name__}."
            )

        # `json.loads` returns `Any`; a JSON object always has string keys, so
        # this cast is the single typed boundary, like `entry_point.load()` in
        # `plugin_discovery`. It is validated above and narrowed nowhere else.
        return cls(cast("dict[str, object]", loaded))

    def required_text(self, key: str) -> str:
        """Return the string value for `key` or raise with a descriptive error."""
        if key not in self.fields:
            raise ValueError(
                f"Missing field {key!r} in tool input. "
                f"Expected a JSON object with a {key!r} string."
            )

        value = self.fields[key]

        if not isinstance(value, str):
            raise ValueError(
                f"Field {key!r} must be a string, got {value!r}. "
                "Expected a JSON string."
            )

        return value
