"""Resolves the real GGUF model at the repo root and skips when it is absent.

Tests run from the package directory (``uv run --directory``), so the model path
is resolved relative to the workspace root, not the current working directory.
"""

from __future__ import annotations

from pathlib import Path

import pytest

MODEL_PATH = Path(__file__).resolve().parents[4] / "models" / "LFM2-8B-A1B-Q4_K_M.gguf"


@pytest.fixture
def model_path() -> Path:
    if not MODEL_PATH.exists():
        pytest.skip(f"Model file not found: {MODEL_PATH}. Skipping integration tests.")
    return MODEL_PATH
