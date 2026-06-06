"""Runs the agent-tools feature under the remote Gemini provider via litellm.

Marked ``network``: skipped (not failed) when ``GEMINI_API_KEY`` is unset. litellm
reads the key from the environment; the shared steps live in ``conftest`` and this
module only supplies the provider-bound ``run_agent`` fixture.
"""

from __future__ import annotations

import time

import pytest
from little_harness.composition import run_cli
from pytest_bdd import scenarios

from tests.e2e.conftest import RunAgent

pytestmark = [pytest.mark.integration, pytest.mark.network]

scenarios("features/agent_tools.feature")


@pytest.fixture
def run_agent(gemini_model: str) -> RunAgent:
    def run(prompt: str, tools: str) -> str:
        start = time.monotonic()
        result = run_cli(
            [
                "--provider",
                "litellm",
                "-o",
                f"model={gemini_model}",
                # Free-tier Gemini caps requests per minute; let LiteLLM ride out
                # the 429s with backoff that honors the server's retry hint.
                "-o",
                "num_retries=8",
                "--tools",
                tools,
                "--prompt",
                prompt,
                "--yes",
                "--max-tokens",
                "512",
                "--max-iterations",
                "4",
            ]
        )
        print(f"\n[perf] {tools}: {time.monotonic() - start:.1f}s", flush=True)
        return result

    return run
