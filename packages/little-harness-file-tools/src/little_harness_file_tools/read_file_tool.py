"""Tool that reads a UTF-8 text file and returns its contents."""

from __future__ import annotations

from pathlib import Path

from little_harness.domain.tool_result import ToolRunRequest, ToolRunResult
from little_harness.domain.tool_spec import ToolExamples, ToolInputSchema, ToolSpec
from little_harness.domain.values.text_values import ToolName, ToolOutput


class ReadFileTool:
    """Returns the text of the file at the given path; reads are safe to run.

    Example:
        ReadFileTool().run(ToolRunRequest(ToolName("read_file"), ToolInput("a.txt")))

    """

    @property
    def spec(self) -> ToolSpec:
        """Return the tool specification for the read_file tool."""
        return ToolSpec(
            ToolName("read_file"),
            "Read a UTF-8 text file and return its full contents.",
            ToolInputSchema(
                "A filesystem path to the file to read.",
                ToolExamples(("README.md", "src/app.py")),
                {"type": "string"},
            ),
        )

    def run(self, request: ToolRunRequest) -> ToolRunResult:
        """Read the file at the path in `request.raw_input` and return its contents."""
        # `read_text()` uses the default text encoding (UTF-8 on the supported
        # platforms). ValueError covers an embedded null byte in the path.
        try:
            contents = Path(request.raw_input.value).read_text()
            return ToolRunResult(
                request.tool_name, ToolOutput(contents), succeeded=True
            )
        except (OSError, ValueError) as error:
            return ToolRunResult(
                request.tool_name,
                ToolOutput(f"Read error: {error}"),
                succeeded=False,
            )
