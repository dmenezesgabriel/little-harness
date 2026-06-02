from __future__ import annotations

from typing import cast

import pytest
from llama_cpp.llama_types import CreateChatCompletionResponse

from local_llm.chat import ChatMessage
from local_llm.llama_cpp_model import extract_response_content, to_llama_message


class TestExtractResponseContent:
    def test_returns_trimmed_first_choice_content(self) -> None:
        # Arrange
        response = create_response(" done ")

        # Act
        content = extract_response_content(response)

        # Assert
        assert content == "done"

    def test_rejects_response_without_choices(self) -> None:
        # Arrange
        response = cast("CreateChatCompletionResponse", {"choices": []})

        # Act / Assert
        with pytest.raises(ValueError, match="Expected at least one response choice"):
            extract_response_content(response)

    def test_rejects_non_string_content(self) -> None:
        # Arrange
        response = create_response(123)

        # Act / Assert
        with pytest.raises(TypeError, match="Expected message content string"):
            extract_response_content(response)


class TestToLlamaMessage:
    @pytest.mark.parametrize(
        ("message", "expected_role"),
        [
            (ChatMessage("system", "rules"), "system"),
            (ChatMessage("assistant", "answer"), "assistant"),
            (ChatMessage("user", "question"), "user"),
        ],
    )
    def test_converts_chat_message_to_llama_message(
        self,
        message: ChatMessage,
        expected_role: str,
    ) -> None:
        # Act
        llama_message = to_llama_message(message)

        # Assert
        assert llama_message["role"] == expected_role
        assert llama_message.get("content") == message.content


def create_response(content: object) -> CreateChatCompletionResponse:
    return cast(
        "CreateChatCompletionResponse",
        {
            "choices": [
                {
                    "message": {
                        "content": content,
                    },
                },
            ],
        },
    )
