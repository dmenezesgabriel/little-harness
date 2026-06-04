"""Tool that writes text to a file, creating parent directories as needed."""

from __future__ import annotations

from pathlib import Path

from little_harness.domain.json_object_input import JsonObjectInput
from little_harness.domain.tool_result import ToolRunRequest, ToolRunResult
from little_harness.domain.tool_spec import ToolExamples, ToolInputSchema, ToolSpec
from little_harness.domain.values.text_values import ToolName, ToolOutput


class WriteFileTool:
    """Writes `content` to `path` (overwriting), creating missing parent dirs.

    Sensitive: it overwrites files, so the runtime asks for approval first.

    Example:
        WriteFileTool().run(request)  # request.raw_input = '{"path","content"}'
    """

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            ToolName("write_file"),
            "Write text to a file, creating parent directories and overwriting.",
            ToolInputSchema(
                'A JSON object {"path": "...", "content": "..."}.',
                ToolExamples(('{"path": "notes.txt", "content": "hello"}',)),
            ),
            requires_approval=True,
        )

    def run(self, request: ToolRunRequest) -> ToolRunResult:
        try:
            return self._write(request)
        except (OSError, ValueError) as error:
            return ToolRunResult(
                request.tool_name,
                ToolOutput(f"Write error: {error}"),
                succeeded=False,
            )

    def _write(self, request: ToolRunRequest) -> ToolRunResult:
        fields = JsonObjectInput.parse(request.raw_input.value)
        path = Path(fields.required_text("path"))
        content = fields.required_text("content")
        path.parent.mkdir(parents=True, exist_ok=True)
        written = path.write_text(content)
        return ToolRunResult(
            request.tool_name,
            ToolOutput(f"Wrote {written} characters to {path}."),
            succeeded=True,
        )
