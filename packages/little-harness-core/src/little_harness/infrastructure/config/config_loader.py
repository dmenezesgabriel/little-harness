"""Read, merge, and resolve TOML config files."""

from __future__ import annotations

import tomllib
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from little_harness.infrastructure.config.config_types import Config

GLOBAL_DIR_NAME = ".little-harness"
CONFIG_FILENAME = "config.toml"

PLUGIN_TOP_KEY = "plugins"
PROFILES_TOP_KEY = "profiles"

FIELD_ALIASES: dict[str, str] = {
    "yes": "approve_all",
}


class ConfigLoader:
    """Reads, merges, and resolves TOML config from global and project paths."""

    def __init__(
        self,
        home_dir: Path | None = None,
        project_root: Path | None = None,
    ) -> None:
        """Initialize with optional paths (defaults to home and cwd)."""
        self._home_dir = home_dir or Path.home()
        self._project_root = project_root or Path.cwd()

    def load(self) -> Config:
        """Load and merge global then project TOML config."""
        global_raw = self._read(self._global_path())
        project_raw = self._read(self._project_path())
        return self._merge_raw(global_raw, project_raw)

    def resolve_profile(self, config: Config, profile_name: str) -> Config:
        """Apply profile values on top of config if the profile exists."""
        profile_values = _find_profile_in_paths(
            profile_name, [self._global_path(), self._project_path()]
        )
        if profile_values is None:
            return config

        return replace(config, **self._raw_to_kwargs(profile_values))

    def _global_path(self) -> Path:
        return self._home_dir / GLOBAL_DIR_NAME / CONFIG_FILENAME

    def _project_path(self) -> Path:
        return self._project_root / GLOBAL_DIR_NAME / CONFIG_FILENAME

    def _read(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        with path.open("rb") as f:
            return _flatten_toml(tomllib.load(f))

    def _merge_raw(self, base: dict[str, Any], overlay: dict[str, Any]) -> Config:
        merged: dict[str, Any] = {}
        merged.update(base)
        merged.update(overlay)

        merged_plugins = dict(base.get("_plugins", {}))
        merged_plugins.update(overlay.get("_plugins", {}))

        merged.pop("_plugins", None)
        merged["plugins"] = merged_plugins

        return Config(**self._raw_to_kwargs(merged))

    def _raw_to_kwargs(self, raw: dict[str, Any]) -> dict[str, Any]:
        kw: dict[str, Any] = {}
        for raw_key, raw_value in raw.items():
            if raw_key.startswith("_"):
                continue
            key = FIELD_ALIASES.get(raw_key, raw_key)
            normalized = _normalize_value(key, raw_value)
            if normalized is not None:
                kw[key] = normalized
        return kw


def _normalize_value(key: str, value: object) -> object:
    if key == "plugins" and isinstance(value, dict):
        plugins_dict = cast(dict[str, Any], value)
        result: dict[str, dict[str, str]] = {}
        for plugin, opts in plugins_dict.items():
            if isinstance(opts, dict):
                opts_dict = cast(dict[str, Any], opts)
                result[str(plugin)] = {str(k): str(v) for k, v in opts_dict.items()}
        return result
    if key == "tools" and isinstance(value, list):
        return tuple(str(v) for v in cast(list[Any], value))
    return value


def _flatten_toml(raw: dict[str, Any]) -> dict[str, Any]:
    """Flatten TOML sections into a single-level dict.

    Preserves [plugins.*] tables as a `_plugins` dict key,
    skips [profiles.*] tables (they're resolved separately).
    """
    result: dict[str, Any] = {}
    plugins: dict[str, dict[str, str]] = {}

    for key, value in raw.items():
        if key == PLUGIN_TOP_KEY and isinstance(value, dict):
            for plugin_name, plugin_opts in cast(dict[str, Any], value).items():
                if isinstance(plugin_opts, dict):
                    opts = cast(dict[str, Any], plugin_opts)
                    plugins[plugin_name] = {k: str(v) for k, v in opts.items()}
            continue
        if key == PROFILES_TOP_KEY:
            continue
        if isinstance(value, dict):
            continue
        result[key] = value

    if plugins:
        result["_plugins"] = plugins

    return result


def _find_profile_in_paths(name: str, paths: list[Path]) -> dict[str, Any] | None:
    for path in paths:
        if not path.exists():
            continue
        with path.open("rb") as f:
            raw = tomllib.load(f)
        profiles = raw.get(PROFILES_TOP_KEY, {})
        if name in profiles and isinstance(profiles[name], dict):
            return profiles[name]
    return None
