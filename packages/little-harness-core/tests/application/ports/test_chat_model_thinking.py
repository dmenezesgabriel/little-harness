"""Tests for thinking fields on ChatCompletionRequest and ChatModel protocol."""

from __future__ import annotations

from collections.abc import Iterator

from little_harness.application.ports.chat_model import (
    ChatCompletionRequest,
    ChatModel,
)
from little_harness.domain.message_history import MessageHistory
from little_harness.domain.values.numeric_values import MaxTokens, Temperature
from little_harness.domain.values.text_values import MessageContent
from little_harness.domain.values.thinking import ThinkingBudget, ThinkingLevel


class TestChatCompletionRequestThinking:
    def test_defaults_to_no_thinking(self) -> None:
        request = ChatCompletionRequest(
            messages=MessageHistory(),
            temperature=Temperature(0.0),
            max_tokens=MaxTokens(128),
        )
        assert request.thinking_level is None
        assert request.thinking_budget is None

    def test_accepts_thinking_level(self) -> None:
        request = ChatCompletionRequest(
            messages=MessageHistory(),
            temperature=Temperature(0.0),
            max_tokens=MaxTokens(128),
            thinking_level=ThinkingLevel.MEDIUM,
        )
        assert request.thinking_level == ThinkingLevel.MEDIUM
        assert request.thinking_budget is None

    def test_accepts_thinking_budget(self) -> None:
        request = ChatCompletionRequest(
            messages=MessageHistory(),
            temperature=Temperature(0.0),
            max_tokens=MaxTokens(128),
            thinking_level=ThinkingLevel.HIGH,
            thinking_budget=ThinkingBudget(2048),
        )
        assert request.thinking_level == ThinkingLevel.HIGH
        assert request.thinking_budget == ThinkingBudget(2048)


class ThinkingChatModel:
    """A ChatModel that advertises thinking support for testing protocols."""

    def supports_thinking(self) -> bool:
        return True

    def complete_streaming(
        self, request: ChatCompletionRequest
    ) -> Iterator[MessageContent]:
        return iter([])

    def close(self) -> None:
        return


class NonThinkingChatModel:
    """A ChatModel that does not advertise thinking support."""

    def supports_thinking(self) -> bool:
        return False

    def complete_streaming(
        self, request: ChatCompletionRequest
    ) -> Iterator[MessageContent]:
        return iter([])

    def close(self) -> None:
        return


class TestChatModelThinkingProtocol:
    def test_thinking_model_reports_support(self) -> None:
        model: ChatModel = ThinkingChatModel()
        assert model.supports_thinking() is True

    def test_non_thinking_model_reports_no_support(self) -> None:
        model: ChatModel = NonThinkingChatModel()
        assert model.supports_thinking() is False
