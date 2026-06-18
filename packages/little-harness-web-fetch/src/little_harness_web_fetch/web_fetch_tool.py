"""HTTP GET tool using stdlib urllib — no external dependencies.

Accepts a JSON object with a required ``url`` string and optional
``format`` (default ``"text"``), ``timeout`` (default 30 seconds).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from little_harness.domain.json_object_input import JsonObjectInput
from little_harness.domain.tool_result import ToolRunRequest, ToolRunResult
from little_harness.domain.tool_spec import ToolExamples, ToolInputSchema, ToolSpec
from little_harness.domain.values.text_values import ToolName, ToolOutput


class WebFetchTool:
    """Fetch content from a URL via HTTP GET.

    Uses stdlib ``urllib.request``. The URL opener is injectable so callers
    can supply a fake for testing.

    Example:
        tool = WebFetchTool()
        result = tool.run(ToolRunRequest(
            ToolName("web_fetch"),
            ToolInput('{"url": "https://example.com", "timeout": 15}'),
        ))

    """

    def __init__(
        self,
        urlopen: Callable[..., Any] = urlopen,
    ) -> None:
        """Accept an injectable URL opener; defaults to stdlib urlopen."""
        self._urlopen = urlopen

    @property
    def spec(self) -> ToolSpec:
        """Return the tool specification with name, description, and schema."""
        return ToolSpec(
            ToolName("web_fetch"),
            "Fetch content from a URL via HTTP GET. "
            "Returns the raw response body as text. "
            "Configure `timeout` (default 30s) for slow servers.",
            ToolInputSchema(
                'A JSON object {"url": "...", "format": "...", "timeout": N}.',
                ToolExamples(
                    (
                        '{"url": "https://example.com"}',
                        '{"url": "https://api.example.com/data", "timeout": 15}',
                    )
                ),
                {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "format": {
                            "type": "string",
                            "enum": ["text", "html"],
                            "default": "text",
                        },
                        "timeout": {"type": "integer", "minimum": 1},
                    },
                    "required": ["url"],
                    "additionalProperties": False,
                },
            ),
        )

    def run(self, request: ToolRunRequest) -> ToolRunResult:
        """Execute the HTTP GET; errors are captured into ToolRunResult, not raised."""
        try:
            return self._execute(request)
        except (HTTPError, URLError, ValueError, OSError) as error:
            return ToolRunResult(
                request.tool_name,
                ToolOutput(f"Web fetch error: {error}"),
                succeeded=False,
            )

    def _execute(self, request: ToolRunRequest) -> ToolRunResult:
        fields = JsonObjectInput.parse(request.raw_input.value)

        url_str = fields.fields.get("url")
        if not url_str or not isinstance(url_str, str):
            raise ValueError(
                f"Field 'url' must be a non-empty string, got {url_str!r};"
                " expected a URL string."
            )

        timeout = 30
        if "timeout" in fields.fields:
            raw = fields.fields["timeout"]
            if not isinstance(raw, int):
                raise ValueError(
                    f"Field 'timeout' must be an integer, got {raw!r}; expected int."
                )
            timeout = raw

        req = Request(url_str, method="GET")
        resp = self._urlopen(req, timeout=timeout)
        body = resp.read().decode("utf-8", errors="replace")

        return ToolRunResult(request.tool_name, ToolOutput(body), succeeded=True)
