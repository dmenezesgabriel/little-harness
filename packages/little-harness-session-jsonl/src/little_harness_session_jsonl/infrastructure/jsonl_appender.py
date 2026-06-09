# ruff: noqa: D100, D101, D102, D103, D107

import json
from pathlib import Path
from typing import Any


class JsonlFileAppender:
    """Safely appends dictionaries as single JSON lines to a file."""

    def __init__(self, file_path: Path) -> None:
        self._file_path = file_path

    def append(self, data: dict[str, Any]) -> None:
        """Serialize data to JSON and append to the file atomically-ish."""
        # Ensure parent directory exists
        self._file_path.parent.mkdir(parents=True, exist_ok=True)  # pragma: no mutate

        json_line = json.dumps(data, separators=(",", ":")) + "\n"  # pragma: no mutate

        with self._file_path.open("a", encoding="utf-8") as f:  # pragma: no mutate
            f.write(json_line)
