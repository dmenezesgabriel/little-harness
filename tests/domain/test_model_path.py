from __future__ import annotations

from pathlib import Path

import pytest

from local_llm.domain.values.model_path import ModelPath


class TestModelPath:
    def test_accepts_a_gguf_path_without_requiring_existence(self) -> None:
        # Arrange
        path = Path("models/does-not-exist.gguf")

        # Act / Assert
        assert ModelPath(path).value == path

    def test_rejects_a_non_gguf_path(self) -> None:
        # Act / Assert
        with pytest.raises(ValueError, match="not a GGUF file"):
            ModelPath(Path("models/model.bin"))
