from __future__ import annotations

from pathlib import Path

import pytest

from local_llm.infrastructure.llama_cpp.chat_model import LlamaCppChatModel
from local_llm.infrastructure.providers.chat_model_factory import (
    LLAMA_CPP_PROVIDER,
    create_chat_model,
)
from tests.infrastructure.llama_cpp.fakes import FakeLlama, make_settings


class TestCreateChatModel:
    def test_builds_the_llama_cpp_model_for_its_provider(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Arrange
        model_file = tmp_path / "model.gguf"
        model_file.write_bytes(b"")
        monkeypatch.setattr(
            "local_llm.infrastructure.llama_cpp.model_factory.Llama", FakeLlama
        )

        # Act
        model = create_chat_model(LLAMA_CPP_PROVIDER, make_settings(model_file))

        # Assert
        assert isinstance(model, LlamaCppChatModel)

    def test_rejects_an_unknown_provider(self) -> None:
        # Act / Assert
        with pytest.raises(ValueError, match="Unknown provider: 'mystery'"):
            create_chat_model("mystery", make_settings(Path("model.gguf")))
