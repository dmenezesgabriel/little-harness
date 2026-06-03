from __future__ import annotations

from pathlib import Path

import pytest

from local_llm.infrastructure.llama_cpp.chat_model import LlamaCppChatModel
from local_llm.infrastructure.providers.chat_model_factory import build_llama_cpp_model
from tests.infrastructure.llama_cpp.fakes import FakeLlama, make_settings


class TestBuildLlamaCppModel:
    def test_builds_a_llama_cpp_chat_model(
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
        model = build_llama_cpp_model(make_settings(model_file))

        # Assert
        assert isinstance(model, LlamaCppChatModel)
