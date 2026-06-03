from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from llama_cpp.llama_types import CreateChatCompletionResponse

from local_llm.application.ports.chat_model import ChatCompletionRequest
from local_llm.domain.message import ChatMessage
from local_llm.domain.message_history import MessageHistory
from local_llm.domain.values.numeric_values import MaxTokens, Temperature
from local_llm.domain.values.role import SYSTEM, USER
from local_llm.domain.values.text_values import MessageContent
from local_llm.infrastructure.llama_cpp.chat_model import (
    LlamaCppChatModel,
    extract_response_content,
)
from tests.infrastructure.llama_cpp.fakes import FakeLlama, make_settings


def make_response(content: object) -> CreateChatCompletionResponse:
    return cast(
        "CreateChatCompletionResponse",
        {"choices": [{"message": {"content": content}}]},
    )


class TestExtractResponseContent:
    def test_returns_trimmed_first_choice_content(self) -> None:
        # Act / Assert
        assert extract_response_content(make_response(" done ")) == "done"

    def test_rejects_response_without_choices(self) -> None:
        # Arrange
        response = cast("CreateChatCompletionResponse", {"choices": []})

        # Act / Assert
        with pytest.raises(ValueError, match="Expected at least one response choice"):
            extract_response_content(response)

    def test_rejects_non_string_content(self) -> None:
        # Act / Assert
        with pytest.raises(TypeError, match="Expected message content string"):
            extract_response_content(make_response(123))


class TestLlamaCppChatModelComplete:
    def test_forwards_request_and_returns_trimmed_content(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Arrange
        model_file = tmp_path / "model.gguf"
        model_file.write_bytes(b"")
        created: list[FakeLlama] = []

        def fake_llama(**kwargs: Any) -> FakeLlama:
            instance = FakeLlama(**kwargs)
            created.append(instance)
            return instance

        monkeypatch.setattr(
            "local_llm.infrastructure.llama_cpp.model_factory.Llama", fake_llama
        )
        chat_model = LlamaCppChatModel(make_settings(model_file))
        request = ChatCompletionRequest(
            MessageHistory()
            .with_message(ChatMessage(SYSTEM, MessageContent("rules")))
            .with_message(ChatMessage(USER, MessageContent("hi"))),
            Temperature(0.2),
            MaxTokens(64),
        )

        # Act
        response = chat_model.complete(request)

        # Assert
        assert response.content == MessageContent("hi there")
        recorded = created[0].completion_kwargs
        assert recorded["temperature"] == request.temperature.value
        assert recorded["max_tokens"] == request.max_tokens.value
        assert [message["role"] for message in recorded["messages"]] == [
            "system",
            "user",
        ]
