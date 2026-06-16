"""Entry point for little-harness-ls. Returns a configured LsTool instance."""

from __future__ import annotations

from little_harness_ls.ls_tool import LsTool


def build() -> LsTool:
    """Return a ready-to-use LsTool instance."""
    return LsTool()
