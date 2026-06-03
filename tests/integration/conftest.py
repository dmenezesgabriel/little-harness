"""Skips the integration suite when the real model file is not available."""

from __future__ import annotations

from pathlib import Path

import pytest

MODEL_PATH = Path("models/LFM2-8B-A1B-Q4_K_M.gguf")


@pytest.fixture(autouse=True)
def require_model_file() -> None:
    if not MODEL_PATH.exists():
        pytest.skip(f"Model file not found: {MODEL_PATH}. Skipping integration tests.")
