"""Tool that searches a file's syntax tree with a tree-sitter query."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from little_harness.domain.json_object_input import JsonObjectInput
from little_harness.domain.tool_result import ToolRunRequest, ToolRunResult
from little_harness.domain.tool_spec import ToolExamples, ToolInputSchema, ToolSpec
from little_harness.domain.values.text_values import ToolName, ToolOutput

from little_harness_ast.structural_match import StructuralMatch
from little_harness_ast.syntax_engine import SyntaxEngine

NO_MATCHES_MESSAGE = "No matches found."
EXAMPLE_INPUT = '{"path": "app.py", "language": "python", "query": "(call) @match"}'


class AstGrepTool:
    """Returns every node a tree-sitter query captures as `@match`; safe to run.

    Example:
        AstGrepTool(engine).run(request)  # raw_input = '{"path","language","query"}'

    """

    def __init__(self, engine: SyntaxEngine) -> None:
        """See class docstring for argument descriptions."""
        self._engine = engine

    @property
    def spec(self) -> ToolSpec:
        """Return the tool's specification: name, description, JSON schema."""
        return ToolSpec(
            ToolName("ast_grep"),
            "Search code structure with tree-sitter query (capture @match).",
            ToolInputSchema(
                'A JSON object {"path": "...", "language": "...", "query": "..."}.',
                ToolExamples((EXAMPLE_INPUT,)),
                {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "language": {"type": "string"},
                        "query": {"type": "string"},
                    },
                    "required": ["path", "language", "query"],
                    "additionalProperties": False,
                },
            ),
        )

    def run(self, request: ToolRunRequest) -> ToolRunResult:
        """Execute an AST grep: parse input, run query, format matches."""
        try:
            return self._search(request)
        except (OSError, ValueError) as error:
            return ToolRunResult(
                request.tool_name,
                ToolOutput(f"ast-grep error: {error}"),
                succeeded=False,
            )

    def _search(self, request: ToolRunRequest) -> ToolRunResult:
        fields = JsonObjectInput.parse(request.raw_input.value)
        path = fields.required_text("path")
        language = fields.required_text("language")
        query = fields.required_text("query")
        source = Path(path).read_text()
        matches = self._engine.find_matches(source, language, query)
        return ToolRunResult(
            request.tool_name, ToolOutput(format_matches(matches, path)), succeeded=True
        )


def format_matches(matches: Sequence[StructuralMatch], path: str) -> str:
    """Render matches as `path:line: text` lines, or a no-match message."""
    if not matches:
        return NO_MATCHES_MESSAGE

    return "\n".join(f"{path}:{match.location()}: {match.text}" for match in matches)
