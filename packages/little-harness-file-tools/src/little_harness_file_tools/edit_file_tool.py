"""Tool that replaces a unique snippet of text within an existing file."""

from __future__ import annotations

from pathlib import Path

from little_harness.domain.json_object_input import JsonObjectInput
from little_harness.domain.tool_result import ToolRunRequest, ToolRunResult
from little_harness.domain.tool_spec import ToolExamples, ToolInputSchema, ToolSpec
from little_harness.domain.values.text_values import ToolName, ToolOutput


class EditFileTool:
    """Replaces the single occurrence of `old` with `new` in a file's text.

    Requiring a unique match keeps edits unambiguous: zero or several matches
    are an error, not a silent guess. Sensitive, so approval is requested.

    Example:
        EditFileTool().run(request)  # raw_input = '{"path","old","new"}'
    """

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            ToolName("edit_file"),
            "Replace the unique occurrence of a text snippet in a file.",
            ToolInputSchema(
                'A JSON object {"path": "...", "old": "...", "new": "..."}.',
                ToolExamples(('{"path": "app.py", "old": "x = 1", "new": "x = 2"}',)),
            ),
            requires_approval=True,
        )

    def run(self, request: ToolRunRequest) -> ToolRunResult:
        try:
            return self._edit(request)
        except (OSError, ValueError) as error:
            return ToolRunResult(
                request.tool_name,
                ToolOutput(f"Edit error: {error}"),
                succeeded=False,
            )

    def _edit(self, request: ToolRunRequest) -> ToolRunResult:
        fields = JsonObjectInput.parse(request.raw_input.value)
        path = Path(fields.required_text("path"))
        original = path.read_text()
        edited = replace_unique(
            original, fields.required_text("old"), fields.required_text("new"), path
        )
        path.write_text(edited)
        return ToolRunResult(
            request.tool_name,
            ToolOutput(f"Replaced 1 occurrence in {path}."),
            succeeded=True,
        )


def replace_unique(text: str, old: str, new: str, path: Path) -> str:
    count = text.count(old)

    if count == 0:
        raise ValueError(
            f"Text to replace not found in {path}: {old!r}. "
            "Expected it to occur exactly once."
        )

    if count > 1:
        raise ValueError(
            f"Text to replace occurs {count} times in {path}: {old!r}. "
            "Expected exactly one occurrence."
        )

    return text.replace(old, new)
