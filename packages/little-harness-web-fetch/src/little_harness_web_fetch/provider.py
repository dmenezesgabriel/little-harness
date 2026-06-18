"""Entry point for little-harness-web-fetch. Returns a configured WebFetchTool."""

from __future__ import annotations

from little_harness_web_fetch.web_fetch_tool import WebFetchTool


def build() -> WebFetchTool:
    """Build a WebFetchTool using the stdlib urlopen backend."""
    return WebFetchTool()
