from __future__ import annotations

from pathlib import Path
from typing import Any

from little_harness.config_types import Config
from little_harness.infrastructure.config.config_loader import ConfigLoader


def write_toml(path: Path, data: dict[str, Any]) -> None:
    """Write a dict as TOML (simplified — values only)."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    _serialize_toml(lines, data, "")
    path.write_text("\n".join(lines))


def _serialize_toml(lines: list[str], data: dict[str, Any], prefix: str) -> None:
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            is_section = any(isinstance(v, dict) for v in value.values())
            if is_section:
                lines.append(f"[{full_key}]")
                _serialize_toml(lines, value, "")
            else:
                _serialize_toml(lines, value, full_key)
        elif isinstance(value, list):
            items = ", ".join(repr(v) for v in value)
            lines.append(f"{key} = [{items}]")
        elif isinstance(value, bool):
            lines.append(f"{key} = {'true' if value else 'false'}")
        elif isinstance(value, str):
            lines.append(f'{key} = "{value}"')
        else:
            lines.append(f"{key} = {value}")


class TestConfigLoader:
    def test_no_files_returns_empty_config(self, tmp_path: Path) -> None:
        loader = ConfigLoader(home_dir=tmp_path, project_root=tmp_path)
        config = loader.load()
        assert config == Config()

    def test_global_config_is_loaded(self, tmp_path: Path) -> None:
        toml_dir = tmp_path / ".little-harness"
        toml_dir.mkdir(parents=True)
        (toml_dir / "config.toml").write_text(
            'temperature = 0.7\nprovider = "llama_cpp"\n'
        )

        loader = ConfigLoader(home_dir=tmp_path, project_root=tmp_path)
        config = loader.load()

        assert config.temperature == 0.7
        assert config.provider == "llama_cpp"

    def test_project_config_overrides_global(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        home_toml = home / ".little-harness" / "config.toml"
        home_toml.parent.mkdir(parents=True)
        home_toml.write_text('temperature = 0.7\nprovider = "llama_cpp"\n')

        project = tmp_path / "project"
        project_toml = project / ".little-harness" / "config.toml"
        project_toml.parent.mkdir(parents=True)
        project_toml.write_text("temperature = 0.5\n")

        loader = ConfigLoader(home_dir=home, project_root=project)
        config = loader.load()

        assert config.temperature == 0.5
        assert config.provider == "llama_cpp"

    def test_loads_plugins_from_config(self, tmp_path: Path) -> None:
        toml_dir = tmp_path / ".little-harness"
        toml_dir.mkdir(parents=True)
        (toml_dir / "config.toml").write_text(
            '[plugins."little-harness-llama-cpp"]\n'
            'model_path = "/models/llama.gguf"\n'
            'n_ctx = "8192"\n'
        )

        loader = ConfigLoader(home_dir=tmp_path, project_root=tmp_path)
        config = loader.load()

        expected = {
            "little-harness-llama-cpp": {
                "model_path": "/models/llama.gguf",
                "n_ctx": "8192",
            }
        }
        assert config.plugins == expected

    def test_merges_plugins_from_global_and_project(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        home_toml = home / ".little-harness" / "config.toml"
        home_toml.parent.mkdir(parents=True)
        home_toml.write_text('[plugins."provider-a"]\nkey1 = "global"\n')

        project = tmp_path / "project"
        project_toml = project / ".little-harness" / "config.toml"
        project_toml.parent.mkdir(parents=True)
        project_toml.write_text(
            '[plugins."provider-a"]\nkey1 = "project"\n'
            '[plugins."provider-b"]\nkey2 = "project"\n'
        )

        loader = ConfigLoader(home_dir=home, project_root=project)
        config = loader.load()

        assert config.plugins["provider-a"]["key1"] == "project"
        assert config.plugins["provider-b"]["key2"] == "project"

    def test_profile_not_applied_by_default(self, tmp_path: Path) -> None:
        toml_dir = tmp_path / ".little-harness"
        toml_dir.mkdir(parents=True)
        (toml_dir / "config.toml").write_text(
            'profile = "fast"\ntemperature = 0.7\n[profiles.fast]\ntemperature = 0.0\n'
        )

        loader = ConfigLoader(home_dir=tmp_path, project_root=tmp_path)
        config = loader.load()

        assert config.profile == "fast"
        assert config.temperature == 0.7  # profile NOT applied yet

    def test_resolve_profile_applies_profile_values(self, tmp_path: Path) -> None:
        toml_dir = tmp_path / ".little-harness"
        toml_dir.mkdir(parents=True)
        (toml_dir / "config.toml").write_text(
            "temperature = 0.7\n"
            "max_tokens = 2048\n"
            "[profiles.fast]\n"
            "temperature = 0.0\n"
            "max_tokens = 1024\n"
        )

        loader = ConfigLoader(home_dir=tmp_path, project_root=tmp_path)
        config = loader.resolve_profile(loader.load(), "fast")

        assert config.temperature == 0.0
        assert config.max_tokens == 1024

    def test_resolve_profile_missing_name_returns_unchanged(
        self, tmp_path: Path
    ) -> None:
        toml_dir = tmp_path / ".little-harness"
        toml_dir.mkdir(parents=True)
        (toml_dir / "config.toml").write_text(
            "temperature = 0.7\n[profiles.fast]\ntemperature = 0.0\n"
        )

        loader = ConfigLoader(home_dir=tmp_path, project_root=tmp_path)
        config = loader.load()
        result = loader.resolve_profile(config, "nonexistent")

        assert result is config  # same object, unchanged

    def test_loads_tools_as_tuple(self, tmp_path: Path) -> None:
        toml_dir = tmp_path / ".little-harness"
        toml_dir.mkdir(parents=True)
        (toml_dir / "config.toml").write_text('tools = ["read_file", "bash"]\n')

        loader = ConfigLoader(home_dir=tmp_path, project_root=tmp_path)
        config = loader.load()

        assert config.tools == ("read_file", "bash")

    def test_loads_yes_as_approve_all(self, tmp_path: Path) -> None:
        toml_dir = tmp_path / ".little-harness"
        toml_dir.mkdir(parents=True)
        (toml_dir / "config.toml").write_text("yes = true\n")

        loader = ConfigLoader(home_dir=tmp_path, project_root=tmp_path)
        config = loader.load()

        assert config.approve_all is True
