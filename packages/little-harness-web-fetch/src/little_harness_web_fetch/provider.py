"""Entry point for little-harness-web-fetch. Returns a configured WebFetchTool."""

from __future__ import annotations

from little_harness_web_fetch.web_fetch_tool import WebFetchTool


def build() -> WebFetchTool:
    return WebFetchTool()
