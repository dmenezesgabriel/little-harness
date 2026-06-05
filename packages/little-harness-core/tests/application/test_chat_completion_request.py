from __future__ import annotations

from little_harness.application.ports.chat_model import (
    ChatCompletionRequest,
    ResponseSchema,
)
from little_harness.domain.message import ChatMessage
from little_harness.domain.message_history import MessageHistory
from little_harness.domain.values.numeric_values import MaxTokens, Temperature
from little_harness.domain.values.role import USER
from little_harness.domain.values.text_values import MessageContent


def one_message_history() -> MessageHistory:
    return MessageHistory().with_message(ChatMessage(USER, MessageContent("hi")))


class TestChatCompletionRequest:
    def test_defaults_to_no_response_schema(self) -> None:
        # Act: a request built without a schema leaves decoding unconstrained.
        request = ChatCompletionRequest(
            one_message_history(), Temperature(0.0), MaxTokens(64)
        )

        # Assert
        assert request.response_schema is None

    def test_carries_a_response_schema_when_given(self) -> None:
        # Arrange
        schema = ResponseSchema({"type": "object"})

        # Act
        request = ChatCompletionRequest(
            one_message_history(), Temperature(0.0), MaxTokens(64), schema
        )

        # Assert
        assert request.response_schema is schema
