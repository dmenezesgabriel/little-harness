from __future__ import annotations

import pytest
from little_harness_litellm.settings import LiteLLMSettings


class TestLiteLLMSettings:
    def test_normalizes_the_model_and_defaults_endpoint_fields(self) -> None:
        # Act
        settings = LiteLLMSettings("  gpt-4o  ")

        # Assert
        assert settings.model == "gpt-4o"
        assert settings.api_base is None
        assert settings.api_key is None
        assert settings.num_retries == 0

    def test_keeps_a_non_negative_retry_count(self) -> None:
        # Act / Assert
        count = 3
        assert LiteLLMSettings("gpt-4o", num_retries=count).num_retries == count

    def test_rejects_a_negative_retry_count(self) -> None:
        # Act / Assert: the message names the offending value.
        with pytest.raises(ValueError, match="LiteLLM num_retries is negative: -1"):
            LiteLLMSettings("gpt-4o", num_retries=-1)

    def test_keeps_endpoint_fields_when_given(self) -> None:
        # Act
        settings = LiteLLMSettings("gpt-4o", api_base="https://p/v1", api_key="sk-x")

        # Assert
        assert settings.api_base == "https://p/v1"
        assert settings.api_key == "sk-x"

    def test_rejects_a_blank_model(self) -> None:
        # Act / Assert
        with pytest.raises(ValueError, match="LiteLLM model is empty"):
            LiteLLMSettings("   ")
