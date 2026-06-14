"""Run a single agent trial and record the outcome.

A trial is one prompt run through ``run_cli`` with a specific tool set and
provider. The result is a structured dataclass capturing pass/fail, wall
time, and the raw output text.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from little_harness.composition import run_cli


@dataclass(frozen=True)
class TrialResult:
    """Outcome of a single agent trial."""

    task_id: str
    tool_name: str
    provider: str
    prompt: str
    succeeded: bool
    output: str
    duration_seconds: float
    error: str | None = None


def build_cli_args(
    prompt: str,
    tools: Sequence[str] | None,
    provider: str,
    provider_options: Sequence[str],
) -> list[str]:
    """Construct the ``run_cli`` argument list."""
    cmd = [
        "--provider",
        provider,
    ]
    for option in provider_options:
        cmd.extend(["-o", option])
    if tools is not None:
        cmd.extend(["--tools", ",".join(tools)])
    cmd.extend(
        [
            "--prompt",
            prompt,
            "--yes",
            "--max-tokens",
            "512",
            "--max-iterations",
            "4",
        ]
    )
    return cmd


def run_trial(
    task_id: str,
    prompt: str,
    provider: str,
    provider_options: Sequence[str],
    tools: Sequence[str] | None = None,
    expected_substring: str | None = None,
) -> TrialResult:
    """Run one trial and return the result.

    When ``tools`` is None every installed tool is available. The trial is
    considered successful when ``expected_substring`` is found in the output
    (or when no expected substring is provided). The tool name is derived
    from ``tools[0]`` (the first and usually only candidate).
    """
    tool_name = tools[0] if tools else "all"
    argv = build_cli_args(
        prompt=prompt,
        tools=tools,
        provider=provider,
        provider_options=provider_options,
    )
    start = time.monotonic()
    try:
        output = run_cli(argv)
        duration = time.monotonic() - start
        succeeded = expected_substring is None or expected_substring in output
        return TrialResult(
            task_id=task_id,
            tool_name=tool_name,
            provider=provider,
            prompt=prompt,
            succeeded=succeeded,
            output=output,
            duration_seconds=duration,
        )
    except Exception as error:
        duration = time.monotonic() - start
        return TrialResult(
            task_id=task_id,
            tool_name=tool_name,
            provider=provider,
            prompt=prompt,
            succeeded=False,
            output="",
            duration_seconds=duration,
            error=str(error),
        )


def resolve_model_path(
    env_var: str = "LITTLE_HARNESS_E2E_MODEL",
    default: str = "LFM2.5-8B-A1B-Q4_K_M.gguf",
) -> Path | None:
    """Return the model path or None when the file is missing."""
    # evaluation/ -> little-harness -> packages -> repo root.
    models_dir = Path(__file__).resolve().parents[3] / "models"
    file_name = __import__("os").environ.get(env_var, default)
    model_path = models_dir / file_name
    return model_path if model_path.exists() else None


DEFAULT_LLAMA_OPTIONS: list[str] = []
"""Set by ``resolve_llama_options()`` at module load or left as default."""


def resolve_llama_options(model_path: Path | None = None) -> list[str] | None:
    """Build llama.cpp provider options, or return None when the model is gone."""
    if model_path is None:
        model_path = resolve_model_path()
    if model_path is None:
        return None
    return [
        f"model_path={model_path}",
        "n_ctx=8192",
        "n_threads=4",
        "n_batch=256",
        "n_gpu_layers=0",
        "flash_attn=false",
        "seed=42",
    ]


def resolve_litellm_options() -> list[str] | None:
    """Build litellm provider options, or return None when the API key is missing."""
    import os  # noqa: PLC0415

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return None
    return [
        "model=gemini/gemini-2.5-flash",
        "num_retries=8",
    ]
