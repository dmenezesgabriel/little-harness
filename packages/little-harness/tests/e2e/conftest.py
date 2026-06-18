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
import sys
from collections.abc import Callable
from io import StringIO
from pathlib import Path

import pytest
from pytest_bdd import given, parsers, then, when

# A provider-bound runner: (prompt, comma-separated tool names | None) -> answer.
# When tools is None every installed tool is available.
RunAgent = Callable[[str, str | None], str]

# A REPL runner: (list of prompts, argv) -> captured stdout.
RunRepl = Callable[[list[str], list[str]], str]

# tests/e2e/conftest.py -> e2e -> tests -> little-harness -> packages -> repo root.
MODELS_DIRECTORY = Path(__file__).resolve().parents[4] / "models"
# The 350M model cannot reliably call tools: it tends to jump to a final answer
# rather than calling the requested tool. The 8B model handles tool calling
# correctly. Override with LITTLE_HARNESS_E2E_MODEL=LFM2.5-350M-Q4_K_M.gguf for
# a fast (unreliable) run.
DEFAULT_LOCAL_MODEL = "LFM2.5-8B-A1B-Q4_K_M.gguf"
DEFAULT_GEMINI_MODEL = "gemini/gemini-2.5-flash"
DEFAULT_LOCAL_CONTEXT_SIZE = "8192"
DEFAULT_LOCAL_THREAD_COUNT = "4"
DEFAULT_LOCAL_BATCH_SIZE = "256"
DEFAULT_LOCAL_GPU_LAYER_COUNT = "0"
DEFAULT_LOCAL_FLASH_ATTENTION = "false"
# Fixed seed so local runs are reproducible at a given temperature.
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


@pytest.fixture
def run_repl() -> RunRepl:
    """Return a callable that drives the REPL with injected stdin/stdout.

    Each prompt in the list is sent as a separate line. An implicit ``/exit``
    is appended so the loop terminates cleanly. The captured stdout text is
    returned for assertion.
    """

    def _run(prompts: list[str], argv: list[str]) -> str:
        inputs = "\n".join(prompts) + "\n/exit\n"
        output = StringIO()
        old_stdin, old_stdout = sys.stdin, sys.stdout
        sys.stdin = StringIO(inputs)
        sys.stdout = output
        try:
            from little_harness.composition import (  # noqa: PLC0415
                run_cli,
            )

            run_cli(argv)
        finally:
            sys.stdin, sys.stdout = old_stdin, old_stdout
        return output.getvalue()

    return _run


@given(parsers.parse('a workspace file "{name}" containing "{content}"'))
def workspace_file(workspace: Path, name: str, content: str) -> None:
    path = workspace / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


@given(parsers.parse('a workspace file "{name}" with text'))
def workspace_file_docstring(workspace: Path, name: str, docstring: str) -> None:
    """Like workspace_file but uses a Gherkin docstring (preserves newlines)."""
    path = workspace / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(docstring, encoding="utf-8")


@when(parsers.parse('the agent is asked to read "{name}"'), target_fixture="answer")
def ask_read_file(run_agent: RunAgent, name: str) -> str:
    return run_agent(
        f"Read the file {name} and tell me what it says.",
        "read_file",
    )


@when(
    parsers.parse('the agent is asked to write "{content}" into "{name}"'),
    target_fixture="answer",
)
def ask_write_file(run_agent: RunAgent, content: str, name: str) -> str:
    return run_agent(
        f'Create a file named "{name}" with the content "{content}". Then say done.',
        "write_file",
    )


@when(
    parsers.parse('the agent is asked to change "{old}" to "{new}" in "{name}"'),
    target_fixture="answer",
)
def ask_edit_file(run_agent: RunAgent, old: str, new: str, name: str) -> str:
    return run_agent(
        f'Change "{old}" to "{new}" in the file {name}. Then say done.',
        "edit_file",
    )


@when(
    parsers.parse('the agent is asked to run a shell command printing "{token}"'),
    target_fixture="answer",
)
def ask_bash(run_agent: RunAgent, token: str) -> str:
    return run_agent(
        f"Run the command printf {token} and tell me the output.",
        "bash",
    )


@when(
    parsers.parse('the agent is asked the arithmetic question "{question}"'),
    target_fixture="answer",
)
def ask_calculator(run_agent: RunAgent, question: str) -> str:
    return run_agent(
        f"What is {question}? Work it out.",
        "calculator",
    )


@when(
    parsers.parse('the agent is asked to search the workspace for "{term}"'),
    target_fixture="answer",
)
def ask_ripgrep(run_agent: RunAgent, term: str) -> str:
    return run_agent(
        f'Search the workspace for "{term}" and show me the matching line.',
        "ripgrep",
    )


@when(
    parsers.parse(
        'the agent is asked to search the workspace including hidden files for "{term}"'
    ),
    target_fixture="answer",
)
def ask_ripgrep_hidden(run_agent: RunAgent, term: str) -> str:
    return run_agent(
        f'Search hidden files too for "{term}" and show the matching line.',
        "ripgrep",
    )


@when(
    parsers.parse('the agent is asked to find print calls in the Python file "{name}"'),
    target_fixture="answer",
)
def ask_ast_grep(run_agent: RunAgent, name: str) -> str:
    return run_agent(
        f"Find all function calls in {name} using AST search with query "
        '"(call) @match". Then show me the matches.',
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
        f"Rename the function {old} to {new} in {name} using AST edit with query "
        '"(function_definition name: (identifier) @match)". Then say done.',
        "ast_edit",
    )


@when(
    parsers.parse(
        'the agent is asked to find files matching "{pattern}" in the workspace'
    ),
    target_fixture="answer",
)
def ask_find(run_agent: RunAgent, pattern: str) -> str:
    return run_agent(
        f"Use the find tool with pattern '{pattern}' (no leading slash, "
        f"relative to current directory) to search for files. "
        "Show me the matching file names.",
        "find",
    )


@when(
    parsers.parse("the agent is asked to list the workspace directory"),
    target_fixture="answer",
)
def ask_list(run_agent: RunAgent) -> str:
    return run_agent(
        "Use the ls tool to list the files in the current directory. "
        "Show me the file names.",
        "ls",
    )


@when(
    parsers.parse('the agent is asked to fetch the URL "{url}"'),
    target_fixture="answer",
)
def ask_web_fetch(run_agent: RunAgent, url: str) -> str:
    return run_agent(
        f"Use the web_fetch tool to retrieve the content of {url}. "
        "Show me the content.",
        "web_fetch",
    )


# -- "with all tools" step variants (Set B) -----------------------------------
# Same prompts as above but tools=None so every installed tool is available,
# testing whether the model selects the correct one from the full set.


@when(
    parsers.parse('the agent with all tools is asked to read "{name}"'),
    target_fixture="answer",
)
def ask_read_file_all(run_agent: RunAgent, name: str) -> str:
    return run_agent(
        f"Read the file {name} and tell me what it says.",
        None,
    )


@when(
    parsers.parse(
        'the agent with all tools is asked to write "{content}" into "{name}"'
    ),
    target_fixture="answer",
)
def ask_write_file_all(run_agent: RunAgent, content: str, name: str) -> str:
    return run_agent(
        f'Create a file named "{name}" with the content "{content}". Then say done.',
        None,
    )


@when(
    parsers.parse(
        'the agent with all tools is asked to change "{old}" to "{new}" in "{name}"'
    ),
    target_fixture="answer",
)
def ask_edit_file_all(run_agent: RunAgent, old: str, new: str, name: str) -> str:
    return run_agent(
        f'Change "{old}" to "{new}" in the file {name}. Then say done.',
        None,
    )


@when(
    parsers.parse(
        'the agent with all tools is asked to run a shell command printing "{token}"'
    ),
    target_fixture="answer",
)
def ask_bash_all(run_agent: RunAgent, token: str) -> str:
    return run_agent(
        f"Run the command printf {token} and tell me the output.",
        None,
    )


@when(
    parsers.parse(
        'the agent with all tools is asked the arithmetic question "{question}"'
    ),
    target_fixture="answer",
)
def ask_calculator_all(run_agent: RunAgent, question: str) -> str:
    return run_agent(
        f"What is {question}? Work it out.",
        None,
    )


@when(
    parsers.parse(
        'the agent with all tools is asked to search the workspace for "{term}"'
    ),
    target_fixture="answer",
)
def ask_ripgrep_all(run_agent: RunAgent, term: str) -> str:
    return run_agent(
        f'Search the workspace for "{term}" and show me the matching line.',
        None,
    )


@when(
    parsers.parse(
        "the agent with all tools is asked to search the workspace including "
        'hidden files for "{term}"'
    ),
    target_fixture="answer",
)
def ask_ripgrep_hidden_all(run_agent: RunAgent, term: str) -> str:
    return run_agent(
        f'Search hidden files too for "{term}" and show the matching line.',
        None,
    )


@when(
    parsers.parse(
        "the agent with all tools is asked to find print calls in the Python "
        'file "{name}"'
    ),
    target_fixture="answer",
)
def ask_ast_grep_all(run_agent: RunAgent, name: str) -> str:
    return run_agent(
        f"Find all function calls in {name} using AST search with query "
        '"(call) @match". Then show me the matches.',
        None,
    )


@when(
    parsers.parse(
        "the agent with all tools is asked to rename the Python function "
        '"{old}" to "{new}" in "{name}"'
    ),
    target_fixture="answer",
)
def ask_ast_edit_all(run_agent: RunAgent, old: str, new: str, name: str) -> str:
    return run_agent(
        f"Rename the function {old} to {new} in {name} using AST edit with query "
        '"(function_definition name: (identifier) @match)". Then say done.',
        None,
    )


@when(
    parsers.parse(
        'the agent with all tools is asked to find files matching "{pattern}" '
        "in the workspace"
    ),
    target_fixture="answer",
)
def ask_find_all(run_agent: RunAgent, pattern: str) -> str:
    return run_agent(
        f"Use the find tool with pattern '{pattern}' (no leading slash, "
        f"relative to current directory) to search for files. "
        "Show me the matching file names.",
        None,
    )


@when(
    parsers.parse("the agent with all tools is asked to list the workspace directory"),
    target_fixture="answer",
)
def ask_list_all(run_agent: RunAgent) -> str:
    return run_agent(
        "Use the ls tool to list the files in the current directory. "
        "Show me the file names.",
        None,
    )


@when(
    parsers.parse('the agent with all tools is asked to fetch the URL "{url}"'),
    target_fixture="answer",
)
def ask_web_fetch_all(run_agent: RunAgent, url: str) -> str:
    return run_agent(
        f"Use the web_fetch tool to retrieve the content of {url}. "
        "Show me the content.",
        None,
    )


@then(parsers.parse('the answer contains "{text}"'))
def answer_contains(answer: str, text: str) -> None:
    assert text in answer, f"expected {text!r} in agent answer, got: {answer!r}"


@then(parsers.parse('the workspace file "{name}" contains "{content}"'))
def workspace_file_contains(workspace: Path, name: str, content: str) -> None:
    actual = (workspace / name).read_text(encoding="utf-8")
    assert content in actual, f"expected {content!r} in {name}, got: {actual!r}"


@when("I run the repl with prompts", target_fixture="repl_output")
def ask_repl(
    docstring: str,
    run_repl: RunRepl,
    local_llama_options: list[str],
) -> str:
    """Run the interactive REPL with prompts from the Gherkin docstring.

    Lines are stripped; blank lines are ignored. An implicit ``/exit`` is
    appended by the ``run_repl`` fixture.
    """
    prompts = [p.strip() for p in docstring.strip().split("\n") if p.strip()]
    provider_options = [
        item for option in local_llama_options for item in ("-o", option)
    ]
    argv = [
        "--provider",
        "llama_cpp",
        *provider_options,
        "--tools",
        "calculator,read_file",
        "--yes",
        "--max-tokens",
        "512",
        "--max-iterations",
        "4",
    ]
    return run_repl(prompts, argv)


@then(parsers.parse('the repl output contains "{text}"'))
def repl_output_contains(repl_output: str, text: str) -> None:
    assert text in repl_output, (
        f"expected {text!r} in repl output, got: {repl_output!r}"
    )
