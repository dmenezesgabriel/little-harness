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
scenarios("features/interactive.feature")
scenarios("features/tool_selection.feature")


@pytest.fixture
def run_agent(local_llama_options: list[str]) -> RunAgent:
    def run(prompt: str, tools: str | None) -> str:
        provider_options = [
            item for option in local_llama_options for item in ("-o", option)
        ]
        cmd = [
            "--provider",
            "llama_cpp",
            *provider_options,
            "--prompt",
            prompt,
            "--yes",
            "--max-tokens",
            "512",
            "--max-iterations",
            "4",
        ]
        if tools is not None:
            cmd.extend(["--tools", tools])
        start = time.monotonic()
        result = run_cli(cmd)
        print(f"\n[perf] {tools}: {time.monotonic() - start:.1f}s", flush=True)
        return result

    return run
