"""TUI themes configuration for little-harness-rich."""

from __future__ import annotations

from textual.theme import Theme

# TokyoNight-inspired color palette for Harness
harness_tokyonight = Theme(
    name="harness-tokyonight",
    primary="#7AA2F7",  # Accent blue
    secondary="#BB9AF7",  # Purple accent
    accent="#2DD4BF",  # Teal accent
    foreground="#C0CAF5",  # High-contrast body text
    background="#11121D",  # Dark blue tint background
    surface="#1A1B2E",  # elevated surface card
    panel="#25283B",  # section background
    warning="#EB8B46",  # Caution amber
    error="#F7768E",  # Error pink
    success="#9ECE6A",  # Success green
    dark=True,
    variables={
        "text-muted": "#545C7E",
    },
)
