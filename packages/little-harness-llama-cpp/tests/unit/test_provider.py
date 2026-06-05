from __future__ import annotations

from pathlib import Path

import pytest
from little_harness_llama_cpp.chat_model import LlamaCppChatModel
from little_harness_llama_cpp.provider import bool_option, build, to_settings
from little_harness_llama_cpp.values import (
    BatchSize,
    ContextSize,
    GpuLayerCount,
    ModelPath,
    ThreadCount,
)

from tests.unit.fakes import FakeLlama


class TestToSettings:
    def test_uses_defaults_when_options_are_empty(self) -> None:
        # Act
        settings = to_settings({})

        # Assert
        assert settings.model_path == ModelPath(Path("models/LFM2-8B-A1B-Q4_K_M.gguf"))
        assert settings.context_size == ContextSize(8192)
        assert settings.thread_count == ThreadCount(8)
        assert settings.gpu_layer_count == GpuLayerCount(0)
        assert settings.batch_size == BatchSize(512)
        assert settings.flash_attention is True

    def test_reads_each_option(self) -> None:
        # Act
        settings = to_settings(
            {
                "model_path": "/tmp/m.gguf",
                "n_ctx": "4096",
                "n_threads": "4",
                "n_gpu_layers": "20",
                "n_batch": "256",
                "flash_attn": "false",
            }
        )

        # Assert
        assert settings.model_path == ModelPath(Path("/tmp/m.gguf"))
        assert settings.context_size == ContextSize(4096)
        assert settings.thread_count == ThreadCount(4)
        assert settings.gpu_layer_count == GpuLayerCount(20)
        assert settings.batch_size == BatchSize(256)
        assert settings.flash_attention is False

    def test_uses_the_generic_model_option_as_the_gguf_path(self) -> None:
        # Act / Assert: --model (the `model` key) is an alias for `model_path`.
        settings = to_settings({"model": "/tmp/from-model.gguf"})
        assert settings.model_path == ModelPath(Path("/tmp/from-model.gguf"))

    def test_prefers_model_path_over_the_generic_model_option(self) -> None:
        # Act
        settings = to_settings(
            {"model_path": "/tmp/explicit.gguf", "model": "/tmp/generic.gguf"}
        )

        # Assert
        assert settings.model_path == ModelPath(Path("/tmp/explicit.gguf"))

    def test_rejects_a_non_integer_option(self) -> None:
        # Act / Assert: the message names the offending key and value.
        with pytest.raises(ValueError, match="Option 'n_ctx' is not an integer"):
            to_settings({"n_ctx": "lots"})


class TestBoolOption:
    def test_uses_the_default_when_the_key_is_absent(self) -> None:
        # Act / Assert: the default is returned untouched, either way.
        assert bool_option({}, "flash_attn", default=True) is True
        assert bool_option({}, "flash_attn", default=False) is False

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("1", True),
            ("true", True),
            ("TRUE", True),
            (" yes ", True),
            ("on", True),
            ("0", False),
            ("false", False),
            ("no", False),
            ("", False),
        ],
    )
    def test_parses_common_truthy_and_falsy_spellings(
        self, raw: str, expected: bool
    ) -> None:
        # Act / Assert: a present value overrides the default both ways.
        assert bool_option({"flash_attn": raw}, "flash_attn", default=False) is expected
        assert bool_option({"flash_attn": raw}, "flash_attn", default=True) is expected


class TestBuild:
    def test_builds_a_llama_cpp_chat_model(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        model_file = tmp_path / "model.gguf"
        model_file.write_bytes(b"")
        monkeypatch.setattr("little_harness_llama_cpp.model_factory.Llama", FakeLlama)

        # Act
        model = build({"model_path": str(model_file)})

        # Assert
        assert isinstance(model, LlamaCppChatModel)
