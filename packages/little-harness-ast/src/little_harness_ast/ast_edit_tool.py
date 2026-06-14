"""Tool that replaces the uniquely matched syntax node with new text.

Unlike a textual edit, the target is selected by a tree-sitter query, so the
edit is structure-aware: it replaces exactly the node captured as `@match`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from little_harness.domain.json_object_input import JsonObjectInput
from little_harness.domain.tool_result import ToolRunRequest, ToolRunResult
from little_harness.domain.tool_spec import ToolExamples, ToolInputSchema, ToolSpec
from little_harness.domain.values.text_values import ToolName, ToolOutput

from little_harness_ast.structural_match import StructuralMatch
from little_harness_ast.syntax_engine import SyntaxEngine

EXAMPLE_INPUT = (
    '{"path": "app.py", "language": "python", '
    '"query": "(call function: (identifier) @_f (#eq? @_f \\"print\\")) @match", '
    '"replacement": "log()"}'
)


class AstEditTool:
    """Replaces the single `@match` node with `replacement`; writes, so sensitive.

    A unique match keeps the edit unambiguous: zero or several matches are an
    error, not a guess.

    Example:
        AstEditTool(engine).run(request)  # '{"path","language","query","replacement"}'

    """

    def __init__(self, engine: SyntaxEngine) -> None:
        """See class docstring for argument descriptions."""
        self._engine = engine

    @property
    def spec(self) -> ToolSpec:
        """Return the tool's specification: name, description, JSON schema."""
        return ToolSpec(
            ToolName("ast_edit"),
            "Replace AST-matched node in a file.",
            ToolInputSchema(
                'JSON: {"path","language","query","replacement"}',
                ToolExamples((EXAMPLE_INPUT,)),
                {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "language": {"type": "string"},
                        "query": {"type": "string"},
                        "replacement": {"type": "string"},
                    },
                    "required": ["path", "language", "query", "replacement"],
                    "additionalProperties": False,
                },
            ),
            requires_approval=True,
        )

    def run(self, request: ToolRunRequest) -> ToolRunResult:
        """Execute an AST edit: parse input, find match, splice replacement."""
        try:
            return self._edit(request)
        except (OSError, ValueError) as error:
            return ToolRunResult(
                request.tool_name,
                ToolOutput(f"ast-edit error: {error}"),
                succeeded=False,
            )

    def _edit(self, request: ToolRunRequest) -> ToolRunResult:
        fields = JsonObjectInput.parse(request.raw_input.value)
        path = Path(fields.required_text("path"))
        language = fields.required_text("language")
        query = fields.required_text("query")
        replacement = fields.required_text("replacement")
        source = path.read_text()
        match = require_single_match(
            self._engine.find_matches(source, language, query), query, path
        )
        path.write_text(splice(source, match, replacement))
        return ToolRunResult(
            request.tool_name,
            ToolOutput(f"Replaced 1 match ({match.location()}) in {path}."),
            succeeded=True,
        )


def require_single_match(
    matches: Sequence[StructuralMatch], query: str, path: Path
) -> StructuralMatch:
    """Return the sole match or raise ValueError (zero or many matches)."""
    if len(matches) == 0:
        raise ValueError(
            f"No match for query {query!r} in {path}. Expected exactly one match."
        )

    if len(matches) > 1:
        raise ValueError(
            f"Query {query!r} matched {len(matches)} nodes in {path}. "
            "Expected exactly one match."
        )

    return matches[0]


def splice(source: str, match: StructuralMatch, replacement: str) -> str:
    """Replace the byte span of `match` in `source` with `replacement`."""
    source_bytes = source.encode()
    edited = (
        source_bytes[: match.start_byte]
        + replacement.encode()
        + source_bytes[match.end_byte :]
    )
    return edited.decode()
