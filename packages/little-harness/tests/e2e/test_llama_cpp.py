"""Runs the agent-tools feature under the real local llama.cpp provider.

Marked ``local_model``: skipped (not failed) when the GGUF is absent. The shared
Given/When/Then steps live in ``conftest``; this module only supplies the
provider-bound ``run_agent`` fixture.
"""

from __future__ import annotations

import time

import pytest
from little_harness.composition import run_cli
from pytest_bdd import scenarios

from tests.e2e.conftest import RunAgent

pytestmark = [pytest.mark.integration, pytest.mark.local_model]

scenarios("features/agent_tools.feature")


@pytest.fixture
def run_agent(local_llama_options: list[str]) -> RunAgent:
    def run(prompt: str, tools: str) -> str:
        provider_options = [
            item for option in local_llama_options for item in ("-o", option)
        ]
        start = time.monotonic()
        result = run_cli(
            [
                "--provider",
                "llama_cpp",
                *provider_options,
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
