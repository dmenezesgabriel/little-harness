from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from llama_cpp.llama_types import CreateChatCompletionStreamResponse

from local_llm.application.ports.chat_model import ChatCompletionRequest
from local_llm.domain.message import ChatMessage
from local_llm.domain.message_history import MessageHistory
from local_llm.domain.values.numeric_values import MaxTokens, Temperature
from local_llm.domain.values.role import SYSTEM, USER
from local_llm.domain.values.text_values import MessageContent
from local_llm.infrastructure.llama_cpp.chat_model import (
    LlamaCppChatModel,
    extract_chunk_content,
)
from tests.infrastructure.llama_cpp.fakes import FakeLlama, make_settings


def make_chunk(delta: dict[str, object]) -> CreateChatCompletionStreamResponse:
    return cast(
        "CreateChatCompletionStreamResponse",
        {"choices": [{"delta": delta}]},
    )


class TestExtractChunkContent:
    def test_returns_content_from_the_delta(self) -> None:
        # Act / Assert
        assert extract_chunk_content(make_chunk({"content": "tok"})) == "tok"

    def test_returns_none_for_a_role_only_delta(self) -> None:
        # Act / Assert
        assert extract_chunk_content(make_chunk({"role": "assistant"})) is None

    def test_returns_none_when_there_are_no_choices(self) -> None:
        # Arrange
        chunk = cast("CreateChatCompletionStreamResponse", {"choices": []})

        # Act / Assert
        assert extract_chunk_content(chunk) is None

    def test_rejects_non_string_content(self) -> None:
        # Act / Assert
        with pytest.raises(TypeError, match="Expected streamed content string"):
            extract_chunk_content(make_chunk({"content": 123}))


class TestLlamaCppChatModelStreaming:
    def test_forwards_request_and_streams_joined_content(
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

        # Act: the role-only first chunk is skipped, the content chunk is yielded.
        chunks = list(chat_model.complete_streaming(request))

        # Assert
        assert "".join(chunk.value for chunk in chunks) == " hi there "
        recorded = created[0].completion_kwargs
        assert recorded["stream"] is True
        assert recorded["temperature"] == request.temperature.value
        assert recorded["max_tokens"] == request.max_tokens.value
        assert [message["role"] for message in recorded["messages"]] == [
            "system",
            "user",
        ]
