"""Pure-Python directory listing tool — no external dependencies.

Follows the AgentTool protocol. Accepts a JSON input with optional ``path``
(defaults to cwd) and optional ``limit`` (defaults to 500). Includes dotfiles.
Appends ``/`` to directory names.
"""

from __future__ import annotations

import os
from pathlib import Path

from little_harness.domain.json_object_input import JsonObjectInput
from little_harness.domain.tool_result import ToolRunRequest, ToolRunResult
from little_harness.domain.tool_spec import ToolExamples, ToolInputSchema, ToolSpec
from little_harness.domain.values.text_values import ToolName, ToolOutput

_DEFAULT_LIMIT = 500


class LsTool:
    """List directory contents.

    Pure-Python implementation using ``os.listdir`` and ``os.stat``.
    Includes dotfiles, appends ``/`` to directory names, sorts entries
    case-insensitively.

    Example:
        tool = LsTool()
        result = tool.run(ToolRunRequest(
            ToolName("ls"),
            ToolInput('{"path": "src", "limit": 50}'),
        ))

    """

    @property
    def spec(self) -> ToolSpec:
        """Return the tool specification for the ls tool."""
        return ToolSpec(
            ToolName("ls"),
            "List files and directories at a given path. "
            "Includes dotfiles. Directories are marked with a trailing `/`. "
            "Output truncates at 500 entries (configure via `limit`).",
            ToolInputSchema(
                'A JSON object {"path": "...", "limit": N}.',
                ToolExamples(
                    (
                        "{}",
                        '{"path": "src"}',
                        '{"path": ".", "limit": 100}',
                    )
                ),
                {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1},
                    },
                    "additionalProperties": False,
                },
            ),
        )

    def run(self, request: ToolRunRequest) -> ToolRunResult:
        """List directory contents and return entry names."""
        try:
            return self._execute(request)
        except (OSError, ValueError) as error:
            return ToolRunResult(
                request.tool_name,
                ToolOutput(f"Ls error: {error}"),
                succeeded=False,
            )

    def _execute(self, request: ToolRunRequest) -> ToolRunResult:
        fields = JsonObjectInput.parse(request.raw_input.value)

        path_str = fields.fields.get("path")
        root = Path(path_str).expanduser().resolve() if path_str else Path.cwd()

        limit = _DEFAULT_LIMIT
        if "limit" in fields.fields:
            limit_raw = fields.fields["limit"]
            if not isinstance(limit_raw, int):
                raise ValueError(f"Field 'limit': expected int, got {limit_raw!r}.")
            limit = limit_raw

        if not root.is_dir():
            raise ValueError(
                f"Not a directory: {root!r}; expected an existing directory."
            )

        entries: list[str] = []
        for name in sorted(os.listdir(root), key=lambda s: s.lower()):
            entry_path = root / name
            try:
                entries.append(f"{name}/" if entry_path.is_dir() else name)
            except OSError:
                continue
            if len(entries) >= limit:
                break

        output = "\n".join(entries) + ("\n" if entries else "")
        return ToolRunResult(request.tool_name, ToolOutput(output), succeeded=True)
