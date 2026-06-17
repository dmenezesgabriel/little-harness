from __future__ import annotations

from little_harness.infrastructure.config.config_types import Config


class TestConfig:
    def test_all_fields_default_to_none(self) -> None:
        config = Config()
        assert config.temperature is None
        assert config.max_tokens is None
        assert config.max_iterations is None
        assert config.top_p is None
        assert config.repeat_penalty is None
        assert config.provider is None
        assert config.model is None
        assert config.policy is None
        assert config.observer is None
        assert config.stream is None
        assert config.tools is None
        assert config.approve_all is None
        assert config.ui is None
        assert config.profile is None

    def test_plugins_defaults_to_empty_dict(self) -> None:
        config = Config()
        assert config.plugins == {}

    def test_is_frozen(self) -> None:
        config = Config(temperature=0.7)
        try:
            config.temperature = 0.5  # type: ignore[misc]
            raise AssertionError("should be frozen")
        except AttributeError:
            pass

    def test_sets_provided_fields(self) -> None:
        config = Config(
            temperature=0.7,
            max_tokens=4096,
            provider="llama_cpp",
            plugins={"my_plugin": {"key": "val"}},
        )
        assert config.temperature == 0.7
        assert config.max_tokens == 4096
        assert config.provider == "llama_cpp"
        assert config.plugins == {"my_plugin": {"key": "val"}}

    def test_tools_is_tuple(self) -> None:
        config = Config(tools=("read_file", "bash"))
        assert config.tools == ("read_file", "bash")
