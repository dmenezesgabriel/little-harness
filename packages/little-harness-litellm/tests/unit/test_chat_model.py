from __future__ import annotations

import pytest
from little_harness.application.ports.chat_model import ChatCompletionRequest
from little_harness.domain.message import ChatMessage
from little_harness.domain.message_history import MessageHistory
from little_harness.domain.values.numeric_values import MaxTokens, Temperature
from little_harness.domain.values.role import SYSTEM, USER
from little_harness.domain.values.text_values import MessageContent
from little_harness_litellm.chat_model import LiteLLMChatModel, extract_delta_content
from little_harness_litellm.settings import LiteLLMSettings

from tests.unit.fakes import (
    ChoiceWithoutDelta,
    DeltaWithoutContent,
    FakeChoice,
    NoChoices,
    NonStreamingCompletion,
    RecordingCompletion,
    chunk_with_choice,
    content_chunk,
    empty_chunk,
)


class TestExtractDeltaContent:
    def test_returns_content_from_the_delta(self) -> None:
        # Act / Assert
        assert extract_delta_content(content_chunk("tok")) == "tok"

    def test_returns_none_when_there_are_no_choices(self) -> None:
        # Act / Assert
        assert extract_delta_content(empty_chunk()) is None

    def test_returns_none_when_the_chunk_has_no_choices_attribute(self) -> None:
        # Act / Assert: a missing attribute is tolerated via the getattr default.
        assert extract_delta_content(NoChoices()) is None

    def test_returns_none_when_the_choice_has_no_delta_attribute(self) -> None:
        # Act / Assert
        assert extract_delta_content(chunk_with_choice(ChoiceWithoutDelta())) is None

    def test_returns_none_when_the_delta_has_no_content_attribute(self) -> None:
        # Act / Assert
        chunk = chunk_with_choice(FakeChoice(DeltaWithoutContent()))
        assert extract_delta_content(chunk) is None

    def test_returns_none_for_a_content_less_delta(self) -> None:
        # Act / Assert
        assert extract_delta_content(content_chunk(None)) is None

    def test_rejects_non_string_content(self) -> None:
        # Act / Assert: the message names the offending type, not a fixed one.
        with pytest.raises(TypeError, match="Expected streamed content string") as err:
            extract_delta_content(content_chunk(123))
        assert "int" in str(err.value)


class TestLiteLLMChatModelStreaming:
    def test_forwards_request_and_streams_joined_content(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange: the first chunk has no content and must be skipped.
        completion = RecordingCompletion(
            [content_chunk(None), content_chunk(" hi "), content_chunk("there")]
        )
        monkeypatch.setattr("litellm.completion", completion)
        model = LiteLLMChatModel(
            LiteLLMSettings("gpt-4o", api_base="https://p/v1", api_key="sk-x")
        )
        request = ChatCompletionRequest(
            MessageHistory()
            .with_message(ChatMessage(SYSTEM, MessageContent("rules")))
            .with_message(ChatMessage(USER, MessageContent("hi"))),
            Temperature(0.2),
            MaxTokens(64),
        )

        # Act
        chunks = list(model.complete_streaming(request))

        # Assert
        assert "".join(chunk.value for chunk in chunks) == " hi there"
        assert completion.kwargs["model"] == "gpt-4o"
        assert completion.kwargs["stream"] is True
        assert completion.kwargs["temperature"] == request.temperature.value
        assert completion.kwargs["max_tokens"] == request.max_tokens.value
        assert completion.kwargs["api_base"] == "https://p/v1"
        assert completion.kwargs["api_key"] == "sk-x"
        assert [message["role"] for message in completion.kwargs["messages"]] == [
            "system",
            "user",
        ]

    def test_rejects_a_non_streaming_response(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange: completion returns a plain object instead of an iterator.
        monkeypatch.setattr("litellm.completion", NonStreamingCompletion())
        model = LiteLLMChatModel(LiteLLMSettings("gpt-4o"))
        request = ChatCompletionRequest(
            MessageHistory().with_message(ChatMessage(USER, MessageContent("hi"))),
            Temperature(0.2),
            MaxTokens(64),
        )

        # Act / Assert: the adapter rejects it and names the offending type.
        with pytest.raises(TypeError, match="Expected a streaming response") as err:
            list(model.complete_streaming(request))
        assert "dict" in str(err.value)

    def test_close_is_a_noop(self) -> None:
        # Act / Assert: LiteLLM has no native resource; close must not raise.
        LiteLLMChatModel(LiteLLMSettings("gpt-4o")).close()
