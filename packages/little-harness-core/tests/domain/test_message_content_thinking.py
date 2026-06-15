"""Tests for MessageContent with optional ThinkingContent."""

from __future__ import annotations

from little_harness.domain.values.text_values import MessageContent
from little_harness.domain.values.thinking import ThinkingContent


class TestMessageContentThinking:
    def test_default_no_thinking(self) -> None:
        content = MessageContent("visible text")
        assert content.value == "visible text"
        assert content.thinking is None

    def test_with_thinking_content(self) -> None:
        thinking = ThinkingContent("Let me reason about this...")
        content = MessageContent("The answer is 12.", thinking=thinking)
        assert content.value == "The answer is 12."
        assert content.thinking is not None
        assert content.thinking.value == "Let me reason about this..."

    def test_plain_text_still_works_backward_compatibly(self) -> None:
        content = MessageContent("hello")
        assert content.value == "hello"
        assert content.thinking is None

    def test_thinking_with_empty_visible_text(self) -> None:
        thinking = ThinkingContent("reasoning only")
        content = MessageContent("", thinking=thinking)
        assert content.value == ""
        assert content.thinking is not None
