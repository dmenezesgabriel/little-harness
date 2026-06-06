"""Shared machinery for the real-provider end-to-end suite.

Both provider modules (``test_llama_cpp`` and ``test_litellm``) bind the same
``agent_tools.feature`` file. The Given/When/Then steps live here so the two
modules stay free of duplication; each module only supplies its own ``run_agent``
fixture (the provider-specific CLI wiring). When steps drive a real model through
``run_cli`` and hand the printed answer to the Then steps via the ``answer``
fixture.

Resolvers skip — never fail — when a prerequisite is missing: the local GGUF for
``local_model`` tests, ``GEMINI_API_KEY`` for ``network`` tests. Run via
``make e2e`` (see the Makefile); a plain ``pytest`` deselects both marker groups.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from pathlib import Path

import pytest
from pytest_bdd import given, parsers, then, when

# A provider-bound runner: (prompt, comma-separated tool names) -> printed answer.
RunAgent = Callable[[str, str], str]

# tests/e2e/conftest.py -> e2e -> tests -> little-harness -> packages -> repo root.
MODELS_DIRECTORY = Path(__file__).resolve().parents[4] / "models"
# The 350M model cannot reliably call tools: it tends to parrot example paths
# from the system prompt rather than following the user instruction.  The 8B
# model handles the JSON tool-calling protocol correctly.  Override with
# LITTLE_HARNESS_E2E_MODEL=LFM2.5-350M-Q4_K_M.gguf for a fast (unreliable) run.
DEFAULT_LOCAL_MODEL = "LFM2.5-8B-A1B-Q4_K_M.gguf"
DEFAULT_GEMINI_MODEL = "gemini/gemini-2.5-flash"
DEFAULT_LOCAL_CONTEXT_SIZE = "8192"
DEFAULT_LOCAL_THREAD_COUNT = "4"
DEFAULT_LOCAL_BATCH_SIZE = "256"
DEFAULT_LOCAL_GPU_LAYER_COUNT = "0"
DEFAULT_LOCAL_FLASH_ATTENTION = "false"
# Fixed seed so local runs are bit-reproducible at temperature 0.
DEFAULT_LOCAL_SEED = "42"


@pytest.fixture
def local_model_path() -> Path:
    """Resolve the local GGUF (override via LITTLE_HARNESS_E2E_MODEL); skip if gone."""
    file_name = os.environ.get("LITTLE_HARNESS_E2E_MODEL", DEFAULT_LOCAL_MODEL)
    model_path = MODELS_DIRECTORY / file_name
    if not model_path.exists():
        pytest.skip(f"Local model not found: {model_path}. Run `make models` first.")
    return model_path


@pytest.fixture
def local_llama_options(local_model_path: Path) -> list[str]:
    """Provider options for quick, reproducible local smoke tests."""
    return [
        f"model_path={local_model_path}",
        option_value("n_ctx", "LITTLE_HARNESS_E2E_N_CTX", DEFAULT_LOCAL_CONTEXT_SIZE),
        option_value(
            "n_threads", "LITTLE_HARNESS_E2E_N_THREADS", DEFAULT_LOCAL_THREAD_COUNT
        ),
        option_value("n_batch", "LITTLE_HARNESS_E2E_N_BATCH", DEFAULT_LOCAL_BATCH_SIZE),
        option_value(
            "n_gpu_layers",
            "LITTLE_HARNESS_E2E_N_GPU_LAYERS",
            DEFAULT_LOCAL_GPU_LAYER_COUNT,
        ),
        option_value(
            "flash_attn",
            "LITTLE_HARNESS_E2E_FLASH_ATTN",
            DEFAULT_LOCAL_FLASH_ATTENTION,
        ),
        option_value("seed", "LITTLE_HARNESS_E2E_SEED", DEFAULT_LOCAL_SEED),
    ]


def option_value(option_name: str, env_name: str, default: str) -> str:
    return f"{option_name}={os.environ.get(env_name, default)}"


@pytest.fixture
def gemini_model() -> str:
    """Resolve the Gemini model id (overridable); skip when no API key is set."""
    if not os.environ.get("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY is unset; skipping remote Gemini e2e tests.")
    return os.environ.get("LITTLE_HARNESS_E2E_GEMINI_MODEL", DEFAULT_GEMINI_MODEL)


@pytest.fixture(autouse=True)
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated CWD for every e2e scenario.

    autouse=True ensures monkeypatch.chdir runs before any step — not just
    scenarios with a Given step that touches the workspace.  Without this,
    write_file (no Given) would write into the pytest start dir instead of
    the tmp workspace and the Then assertion would fail with FileNotFoundError.
    """
    monkeypatch.chdir(tmp_path)
    return tmp_path


@given(parsers.parse('a workspace file "{name}" containing "{content}"'))
def workspace_file(workspace: Path, name: str, content: str) -> None:
    (workspace / name).write_text(content, encoding="utf-8")


@when(parsers.parse('the agent is asked to read "{name}"'), target_fixture="answer")
def ask_read_file(run_agent: RunAgent, name: str) -> str:
    return run_agent(
        f"Call exactly the read_file tool once with input {name}. Then answer with "
        "the file contents.",
        "read_file",
    )


@when(
    parsers.parse('the agent is asked to write "{content}" into "{name}"'),
    target_fixture="answer",
)
def ask_write_file(run_agent: RunAgent, content: str, name: str) -> str:
    return run_agent(
        "Call exactly the write_file tool once with JSON input "
        f'{{"path": "{name}", "content": "{content}"}}. Then answer done.',
        "write_file",
    )


@when(
    parsers.parse('the agent is asked to change "{old}" to "{new}" in "{name}"'),
    target_fixture="answer",
)
def ask_edit_file(run_agent: RunAgent, old: str, new: str, name: str) -> str:
    return run_agent(
        "Call exactly the edit_file tool once with JSON input "
        f'{{"path": "{name}", "old": "{old}", "new": "{new}"}}. '
        "Then answer done.",
        "edit_file",
    )


@when(
    parsers.parse('the agent is asked to run a shell command printing "{token}"'),
    target_fixture="answer",
)
def ask_bash(run_agent: RunAgent, token: str) -> str:
    return run_agent(
        f"Call exactly the bash tool once with input 'printf {token}'. Then answer "
        "with the command output.",
        "bash",
    )


@when(
    parsers.parse('the agent is asked the arithmetic question "{question}"'),
    target_fixture="answer",
)
def ask_calculator(run_agent: RunAgent, question: str) -> str:
    return run_agent(
        f"Call exactly the calculator tool once to solve: {question} Then answer "
        "with the number.",
        "calculator",
    )


@when(
    parsers.parse('the agent is asked to search the workspace for "{term}"'),
    target_fixture="answer",
)
def ask_ripgrep(run_agent: RunAgent, term: str) -> str:
    if shutil.which("rg") is None:
        pytest.skip("rg (ripgrep) is not installed; skipping ripgrep e2e scenario.")
    return run_agent(
        f"Call exactly the ripgrep tool once with input '{term} .'. Then answer "
        "with the matching line.",
        "ripgrep",
    )


@when(
    parsers.parse('the agent is asked to find print calls in the Python file "{name}"'),
    target_fixture="answer",
)
def ask_ast_grep(run_agent: RunAgent, name: str) -> str:
    return run_agent(
        "Call exactly the ast_grep tool once with JSON input "
        f'{{"path": "{name}", "language": "python", "query": "(call) @match"}}. '
        "Then answer with the match.",
        "ast_grep",
    )


@when(
    parsers.parse(
        'the agent is asked to rename the Python function "{old}" to "{new}" '
        'in "{name}"'
    ),
    target_fixture="answer",
)
def ask_ast_edit(run_agent: RunAgent, old: str, new: str, name: str) -> str:
    return run_agent(
        "Call exactly the ast_edit tool once with JSON input "
        f'{{"path": "{name}", "language": "python", '
        f'"query": "(function_definition name: (identifier) @match)", '
        f'"replacement": "{new}"}}. Then answer done. The old name is {old}.',
        "ast_edit",
    )


@then(parsers.parse('the answer contains "{text}"'))
def answer_contains(answer: str, text: str) -> None:
    assert text in answer, f"expected {text!r} in agent answer, got: {answer!r}"


@then(parsers.parse('the workspace file "{name}" contains "{content}"'))
def workspace_file_contains(workspace: Path, name: str, content: str) -> None:
    actual = (workspace / name).read_text(encoding="utf-8")
    assert content in actual, f"expected {content!r} in {name}, got: {actual!r}"
