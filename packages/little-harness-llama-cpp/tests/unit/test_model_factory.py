# pyright: reportPrivateUsage=false, reportArgumentType=false
from __future__ import annotations

import inspect
from pathlib import Path
from typing import cast

import pytest
from little_harness_llama_cpp.model_factory import (
    _formatter_init_without_generation_tags,
    create_llama_model,
)

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
            "seed": 42,
            "n_ctx": 8192,
            "n_threads": 8,
            "n_threads_batch": 8,
            "n_batch": 512,
            "n_gpu_layers": 0,
            "flash_attn": True,
            "verbose": False,
        }


class TestGenerationTagPatch:
    """Integration: the module-level monkey-patch delegates to the sanitizer."""

    def test_patched_init_delegates_to_original(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        recorded: list[str] = []

        def fake_original_init(
            _self: object,
            template: str,
            _eos: str,
            _bos: str,
            *_args: object,
            **_kwargs: object,
        ) -> None:
            recorded.append(template)

        monkeypatch.setattr(
            "little_harness_llama_cpp.model_factory._original_formatter_init",
            fake_original_init,
        )

        # Act
        _formatter_init_without_generation_tags(
            object(),
            "{% generation %}hello{% endgeneration %}",
            "</s>",
            "<s>",
        )

        # Assert
        assert recorded == ["hello"]

    def test_patched_init_passes_unknown_template_through(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        recorded: list[str] = []

        def fake_original_init(
            _self: object,
            template: str,
            _eos: str,
            _bos: str,
            *_args: object,
            **_kwargs: object,
        ) -> None:
            recorded.append(template)

        monkeypatch.setattr(
            "little_harness_llama_cpp.model_factory._original_formatter_init",
            fake_original_init,
        )

        # Act
        _formatter_init_without_generation_tags(
            object(),
            "Hello {{ name }}",
            "</s>",
            "<s>",
        )

        # Assert
        assert recorded == ["Hello {{ name }}"]

    def test_patched_init_passes_self_through(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        recorded_self: list[object] = []

        def recording_init(
            _self: object,
            _template: str,
            _eos: str,
            _bos: str,
            *_args: object,
            **_kwargs: object,
        ) -> None:
            recorded_self.append(_self)

        monkeypatch.setattr(
            "little_harness_llama_cpp.model_factory._original_formatter_init",
            recording_init,
        )
        expected_self = object()

        # Act
        _formatter_init_without_generation_tags(
            expected_self,
            "{% generation %}x{% endgeneration %}",
            "</s>",
            "<s>",
        )

        # Assert
        assert recorded_self == [expected_self]

    def test_patched_init_default_add_generation_prompt_is_true(
        self,
    ) -> None:
        # Act
        sig = inspect.signature(_formatter_init_without_generation_tags)

        # Assert
        param = sig.parameters["add_generation_prompt"]
        assert param.default is True

    def test_patched_init_default_stop_token_ids_is_none(
        self,
    ) -> None:
        # Act
        sig = inspect.signature(_formatter_init_without_generation_tags)

        # Assert
        param = sig.parameters["stop_token_ids"]
        assert param.default is None

    def test_patched_init_carries_all_original_parameters(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        recorded: dict[str, object] = {}

        def recording_init(
            _self: object,
            template: str,
            eos_token: str,
            bos_token: str,
            add_generation_prompt: bool = True,
            stop_token_ids: list[int] | None = None,
        ) -> None:
            recorded["template"] = template
            recorded["eos_token"] = eos_token
            recorded["bos_token"] = bos_token
            recorded["add_generation_prompt"] = add_generation_prompt
            recorded["stop_token_ids"] = stop_token_ids

        monkeypatch.setattr(
            "little_harness_llama_cpp.model_factory._original_formatter_init",
            recording_init,
        )

        # Act
        _formatter_init_without_generation_tags(
            object(),
            "{% generation %}test{% endgeneration %}",
            "</s>",
            "<s>",
            add_generation_prompt=False,
            stop_token_ids=[2],
        )

        # Assert
        assert recorded["template"] == "test"
        assert recorded["eos_token"] == "</s>"
        assert recorded["bos_token"] == "<s>"
        assert recorded["add_generation_prompt"] is False
        assert recorded["stop_token_ids"] == [2]
