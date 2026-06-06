from __future__ import annotations

import pytest
from little_harness.application.ports.chat_model import ChatCompletionRequest
from little_harness.domain.message import ChatMessage
from little_harness.domain.message_history import MessageHistory
from little_harness.domain.values.numeric_values import MaxTokens, Temperature
from little_harness.domain.values.role import USER
from little_harness.domain.values.text_values import MessageContent
from little_harness_litellm.chat_model import LiteLLMChatModel
from little_harness_litellm.provider import build, to_settings

from tests.unit.fakes import RecordingCompletion, content_chunk

MODEL_REQUIRED_MESSAGE = (
    "Option 'model' is required for the litellm provider. "
    "Expected e.g. -o model=gpt-4o."
)


def a_request() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        MessageHistory().with_message(ChatMessage(USER, MessageContent("hi"))),
        Temperature(0.0),
        MaxTokens(16),
    )


class TestToSettings:
    def test_reads_model_and_endpoint_options(self) -> None:
        # Act
        settings = to_settings(
            {"model": "gpt-4o", "api_base": "https://p/v1", "api_key": "sk-x"}
        )

        # Assert
        assert settings.model == "gpt-4o"
        assert settings.api_base == "https://p/v1"
        assert settings.api_key == "sk-x"

    def test_defaults_retries_to_zero(self) -> None:
        # Act / Assert: omitting the option leaves retries disabled.
        assert to_settings({"model": "gpt-4o"}).num_retries == 0

    def test_reads_the_num_retries_option(self) -> None:
        # Act / Assert
        count = 5
        settings = to_settings({"model": "gpt-4o", "num_retries": str(count)})
        assert settings.num_retries == count

    def test_rejects_a_non_integer_num_retries(self) -> None:
        # Act / Assert: the message names the offending key and value.
        with pytest.raises(ValueError, match="Option 'num_retries' is not an integer"):
            to_settings({"model": "gpt-4o", "num_retries": "lots"})

    def test_requires_a_model_option(self) -> None:
        # Act / Assert: the exact message names the missing option and an example.
        with pytest.raises(ValueError) as err:
            to_settings({})
        assert str(err.value) == MODEL_REQUIRED_MESSAGE


class TestBuild:
    def test_builds_a_model_that_uses_the_configured_settings(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        completion = RecordingCompletion([content_chunk("ok")])
        monkeypatch.setattr("litellm.completion", completion)

        # Act
        model = build({"model": "gpt-4o"})
        list(model.complete_streaming(a_request()))

        # Assert: build wires the parsed settings into a working adapter.
        assert isinstance(model, LiteLLMChatModel)
        assert completion.kwargs["model"] == "gpt-4o"
