"""Pure-Python glob search tool — no external dependencies.

Follows the AgentTool protocol. Accepts a JSON input with ``pattern``,
optional ``path`` (defaults to cwd), and optional ``limit`` (defaults to
1000). Skip ``.git/`` and ``node_modules/`` by default.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

from little_harness.domain.json_object_input import JsonObjectInput
from little_harness.domain.tool_result import ToolRunRequest, ToolRunResult
from little_harness.domain.tool_spec import ToolExamples, ToolInputSchema, ToolSpec
from little_harness.domain.values.text_values import ToolName, ToolOutput

_DEFAULT_LIMIT = 1000
_IGNORED_DIRS = frozenset({".git", "node_modules"})


class FindTool:
    """Search for files matching a glob pattern.

    Pure-Python implementation using ``glob.glob``. Ignores ``.git`` and
    ``node_modules`` directories by default.

    Example:
        tool = FindTool()
        result = tool.run(ToolRunRequest(
            ToolName("find"),
            ToolInput('{"pattern": "*.py", "limit": 10}'),
        ))
    """

    @property
    def spec(self) -> ToolSpec:
        """Return the tool specification for the find tool."""
        return ToolSpec(
            ToolName("find"),
            "Search for files matching a glob pattern. "
            "Output truncates at 1000 results (configure via `limit`). "
            "Skips .git/ and node_modules/ automatically.",
            ToolInputSchema(
                'A JSON object {"pattern": "...", "path": "...", "limit": N}.',
                ToolExamples((
                    '{"pattern": "*.py"}',
                    '{"pattern": "**/*.md", "path": "docs", "limit": 50}',
                )),
                {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "path": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1},
                    },
                    "required": ["pattern"],
                    "additionalProperties": False,
                },
            ),
        )

    def run(self, request: ToolRunRequest) -> ToolRunResult:
        """Run a glob search and return matching file paths."""
        try:
            return self._execute(request)
        except (OSError, ValueError) as error:
            return ToolRunResult(
                request.tool_name,
                ToolOutput(f"Find error: {error}"),
                succeeded=False,
            )

    def _execute(self, request: ToolRunRequest) -> ToolRunResult:
        fields = JsonObjectInput.parse(request.raw_input.value)
        pattern = fields.required_text("pattern")

        path_str = fields.fields.get("path")
        root = Path(path_str).expanduser().resolve() if path_str else Path.cwd()

        limit = _DEFAULT_LIMIT
        if "limit" in fields.fields:
            limit_raw = fields.fields["limit"]
            if not isinstance(limit_raw, int):
                raise ValueError(
                    f"Field 'limit' must be an integer, got {limit_raw!r}."
                )
            limit = limit_raw

        if not root.is_dir():
            raise ValueError(
                f"Path is not a directory or does not exist: {root}"
            )

        matches: list[str] = []
        full_pattern = str(root / pattern)
        for entry in sorted(glob.glob(full_pattern, recursive=True)):
            entry_path = Path(entry)
            if not entry_path.is_file():
                continue
            if _is_ignored(entry_path, root):
                continue
            relative = os.path.relpath(entry_path, root)
            matches.append(relative)
            if len(matches) >= limit:
                break

        output = "\n".join(matches) + ("\n" if matches else "")
        return ToolRunResult(
            request.tool_name, ToolOutput(output), succeeded=True
        )


def _is_ignored(entry: Path, root: Path) -> bool:
    """Return True if the entry lives under an ignored directory."""
    for parent in entry.parents:
        if parent == root:
            break
        if parent.name in _IGNORED_DIRS:
            return True
    return False
