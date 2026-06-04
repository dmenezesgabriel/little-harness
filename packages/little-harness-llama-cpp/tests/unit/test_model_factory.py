from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from little_harness_llama_cpp.model_factory import create_llama_model

from tests.unit.fakes import FakeLlama, make_settings


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
        monkeypatch.setattr("little_harness_llama_cpp.model_factory.Llama", FakeLlama)
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
