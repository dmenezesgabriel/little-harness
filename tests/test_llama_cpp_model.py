from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from llama_cpp.llama_types import CreateChatCompletionResponse

from local_llm.chat import ChatCompletionRequest, ChatMessage
from local_llm.llama_cpp_model import (
    LlamaCppChatModel,
    LlamaCppModelSettings,
    create_llama_model,
    extract_response_content,
    to_llama_message,
)


class FakeLlama:
    """Stand-in for llama_cpp.Llama that records constructor and call arguments."""

    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self.completion_kwargs: dict[str, Any] = {}

    def create_chat_completion(
        self,
        *,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
    ) -> CreateChatCompletionResponse:
        self.completion_kwargs = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        return cast(
            "CreateChatCompletionResponse",
            {"choices": [{"message": {"content": " hi there "}}]},
        )


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


class TestCreateLlamaModel:
    def test_rejects_missing_model_file(self, tmp_path: Path) -> None:
        # Arrange
        settings = make_settings(tmp_path / "missing.gguf")

        # Act / Assert
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            create_llama_model(settings)

    def test_passes_settings_to_llama_constructor(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Arrange
        model_file = tmp_path / "model.gguf"
        model_file.write_bytes(b"")
        monkeypatch.setattr("local_llm.llama_cpp_model.Llama", FakeLlama)
        settings = make_settings(model_file)

        # Act
        model = cast("FakeLlama", create_llama_model(settings))

        # Assert
        assert model.init_kwargs == {
            "model_path": str(model_file),
            "n_ctx": 8192,
            "n_threads": 8,
            "n_gpu_layers": 0,
            "verbose": False,
        }


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

        monkeypatch.setattr("local_llm.llama_cpp_model.Llama", fake_llama)
        chat_model = LlamaCppChatModel(make_settings(model_file))
        request = ChatCompletionRequest(
            messages=(ChatMessage("system", "rules"), ChatMessage("user", "hi")),
            temperature=0.2,
            max_tokens=64,
        )

        # Act
        response = chat_model.complete(request)

        # Assert
        assert response.content == "hi there"
        recorded = created[0].completion_kwargs
        assert recorded["temperature"] == request.temperature
        assert recorded["max_tokens"] == request.max_tokens
        assert [message["role"] for message in recorded["messages"]] == [
            "system",
            "user",
        ]


def make_settings(model_path: Path) -> LlamaCppModelSettings:
    return LlamaCppModelSettings(
        model_path=model_path,
        context_size=8192,
        thread_count=8,
        gpu_layer_count=0,
    )


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
