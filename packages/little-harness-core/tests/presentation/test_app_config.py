from __future__ import annotations

from little_harness.domain.values.numeric_values import (
    MaxIterations,
    MaxTokens,
    Temperature,
)
from little_harness.presentation.cli.app_config import AppConfig


class TestAppConfig:
    def test_profile_defaults_to_none(self) -> None:
        assert (
            AppConfig(
                temperature=Temperature(0.1),
                max_tokens=MaxTokens(512),
                max_iterations=MaxIterations(5),
            ).profile
            is None
        )

    def test_profile_can_be_set(self) -> None:
        assert (
            AppConfig(
                profile="fast",
                temperature=Temperature(0.1),
                max_tokens=MaxTokens(512),
                max_iterations=MaxIterations(5),
            ).profile
            == "fast"
        )

    def test_plugin_configs_defaults_to_empty_dict(self) -> None:
        assert (
            AppConfig(
                temperature=Temperature(0.1),
                max_tokens=MaxTokens(512),
                max_iterations=MaxIterations(5),
            ).plugin_configs
            == {}
        )

    def test_plugin_configs_can_be_set(self) -> None:
        configs = {"llama_cpp": {"n_ctx": "8192"}}
        result = AppConfig(
            plugin_configs=configs,
            temperature=Temperature(0.1),
            max_tokens=MaxTokens(512),
            max_iterations=MaxIterations(5),
        )
        assert result.plugin_configs == configs
