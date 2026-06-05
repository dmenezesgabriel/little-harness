from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from little_harness.application.ports.chat_model import (
    ChatCompletionRequest,
    ResponseSchema,
)
from little_harness.domain.message import ChatMessage
from little_harness.domain.message_history import MessageHistory
from little_harness.domain.values.numeric_values import MaxTokens, Temperature
from little_harness.domain.values.role import SYSTEM, USER
from little_harness.domain.values.text_values import MessageContent
from little_harness_llama_cpp.chat_model import (
    LlamaCppChatModel,
    extract_chunk_content,
    to_response_format,
)
from llama_cpp.llama_types import CreateChatCompletionStreamResponse

from tests.unit.fakes import (
    FakeLlama,
    NonStreamingLlama,
    make_settings,
)


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
        # Act / Assert: the message names the offending type, not a fixed one.
        with pytest.raises(TypeError, match="Expected streamed content string") as err:
            extract_chunk_content(make_chunk({"content": 123}))
        assert "int" in str(err.value)


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

        monkeypatch.setattr("little_harness_llama_cpp.model_factory.Llama", fake_llama)
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
        # No schema on the request leaves decoding unconstrained.
        assert recorded["response_format"] is None

    def test_forwards_a_response_schema_as_a_json_grammar(
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

        monkeypatch.setattr("little_harness_llama_cpp.model_factory.Llama", fake_llama)
        chat_model = LlamaCppChatModel(make_settings(model_file))
        schema = {"type": "object", "required": ["action"]}
        request = ChatCompletionRequest(
            MessageHistory().with_message(ChatMessage(USER, MessageContent("hi"))),
            Temperature(0.2),
            MaxTokens(64),
            ResponseSchema(schema),
        )

        # Act
        list(chat_model.complete_streaming(request))

        # Assert: the schema becomes llama.cpp's json_object grammar request.
        assert created[0].completion_kwargs["response_format"] == {
            "type": "json_object",
            "schema": schema,
        }

    def test_close_releases_the_native_model(
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

        monkeypatch.setattr("little_harness_llama_cpp.model_factory.Llama", fake_llama)
        chat_model = LlamaCppChatModel(make_settings(model_file))

        # Act
        chat_model.close()

        # Assert
        assert created[0].closed is True

    def test_rejects_a_non_streaming_response(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Arrange: the SDK returns a plain response object instead of an iterator.
        model_file = tmp_path / "model.gguf"
        model_file.write_bytes(b"")
        monkeypatch.setattr(
            "little_harness_llama_cpp.model_factory.Llama", NonStreamingLlama
        )
        chat_model = LlamaCppChatModel(make_settings(model_file))
        request = ChatCompletionRequest(
            MessageHistory().with_message(ChatMessage(USER, MessageContent("hi"))),
            Temperature(0.2),
            MaxTokens(64),
        )

        # Act / Assert: the adapter rejects it and names the offending type.
        with pytest.raises(TypeError, match="Expected a streaming response") as err:
            list(chat_model.complete_streaming(request))
        assert "dict" in str(err.value)


class TestToResponseFormat:
    def test_returns_none_for_no_schema(self) -> None:
        # Act / Assert
        assert to_response_format(None) is None

    def test_wraps_the_schema_as_a_json_object_grammar(self) -> None:
        # Arrange
        schema = {"type": "object", "required": ["action"]}

        # Act / Assert
        assert to_response_format(ResponseSchema(schema)) == {
            "type": "json_object",
            "schema": schema,
        }
