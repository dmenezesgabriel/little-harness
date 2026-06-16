"""End-to-end tests for session auto-save, resume, and isolation.

Each scenario verifies a real behavioural guarantee of the session system
by driving a real language model through ``run_cli``:

- **Auto-save**: The interactive REPL writes every event to a JSONL file on
  disk (no explicit ``--session`` flag needed — interactive mode always
  persists).
- **Resume**: Starting a new process with ``--session <id>`` loads the prior
  conversation so the model sees its full context.
- **Isolation**: Two different session IDs produce independent contexts;
  writing to one does not leak into the other.
"""

from __future__ import annotations

import sys
import time
import uuid
from io import StringIO
from pathlib import Path

import pytest
from little_harness.composition import run_cli

pytestmark = [pytest.mark.integration, pytest.mark.local_model]


@pytest.fixture(autouse=True)
def session_dir(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated session storage so tests don't share state on disk.

    Overrides ``LITTLE_HARNESS_SESSION_DIR`` to point into the per-test
    tmp directory created by the ``workspace`` autouse fixture.
    """
    sessions_path = workspace / ".little-harness" / "sessions"
    sessions_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LITTLE_HARNESS_SESSION_DIR", str(sessions_path))
    return sessions_path


# ---------------------------------------------------------------------------
# Helpers  (avoid importing the conftest BDD fixtures to keep this module
# self-contained and independently selectable)
# ---------------------------------------------------------------------------


def _run_agent(
    prompt: str,
    local_llama_options: list[str],
    session_id: str | None = None,
) -> str:
    """Run a single one-shot agent turn, optionally inside a session.

    Returns the rendered answer text.
    """
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
        "3",
    ]
    if session_id is not None:
        cmd.extend(["--session", session_id])
    start = time.monotonic()
    result = run_cli(cmd)
    print(f"\n[perf] agent: {time.monotonic() - start:.1f}s", flush=True)
    return result


def _run_repl(
    prompts: list[str],
    local_llama_options: list[str],
    session_id: str | None = None,
) -> str:
    """Drive the interactive REPL with injected stdin/stdout.

    An implicit ``/exit`` is appended so the loop terminates cleanly.
    Returns the full captured stdout.
    """
    provider_options = [
        item for option in local_llama_options for item in ("-o", option)
    ]
    argv = [
        "--provider",
        "llama_cpp",
        *provider_options,
        "--tools",
        "read_file",
        "--yes",
        "--max-tokens",
        "512",
        "--max-iterations",
        "3",
    ]
    if session_id is not None:
        argv.extend(["--session", session_id])

    inputs = "\n".join(prompts) + "\n/exit\n"
    output = StringIO()
    old_stdin, old_stdout = sys.stdin, sys.stdout
    sys.stdin = StringIO(inputs)
    sys.stdout = output
    try:
        run_cli(argv)
    finally:
        sys.stdin, sys.stdout = old_stdin, old_stdout
    return output.getvalue()


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


class TestInteractiveSessionAutoSave:
    """The interactive REPL always saves session events to disk."""

    def test_session_file_written_on_first_turn(
        self,
        local_llama_options: list[str],
        session_dir: Path,
    ) -> None:
        """A JSONL file appears in the session dir after the first REPL turn.

        Interactive mode always builds a session plugin (no ``--session``
        flag required).  After a single user-assistant exchange the file
        must exist and contain at least one event.
        """
        _run_repl(
            ["Say 'hello' and nothing else."],
            local_llama_options,
        )

        jsonl_files = list(session_dir.glob("*.jsonl"))
        assert len(jsonl_files) == 1, (
            f"Expected exactly one session file, found {len(jsonl_files)}: "
            f"{[f.name for f in jsonl_files]}"
        )

        with jsonl_files[0].open(encoding="utf-8") as f:
            lines = [line for line in f if line.strip()]
        assert len(lines) >= 1, "Session file was created but contains no events"

    def test_interactive_resume_preserves_context(
        self,
        local_llama_options: list[str],
        session_dir: Path,
    ) -> None:
        """Starting a new REPL with ``--session`` sees the prior conversation.

        This exercises the full resume path:
          1. Session plugin discovery & build
          2. Observer wiring (session observer replaces the configured one)
          3. Event serialisation to JSONL
          4. History reconstruction from stored events
          5. InteractiveConsole ``_initial_messages`` pre-population
        """
        session_id = f"repl-resume-{uuid.uuid4().hex[:8]}"

        # First REPL: plant information, then exit.
        _run_repl(
            ["Remember that the secret code is 8675309. Reply with just 'ok'."],
            local_llama_options,
            session_id=session_id,
        )

        # Second REPL: retrieve the fact without re-stating it.
        output = _run_repl(
            ["What is the secret code I told you? Answer with just the number."],
            local_llama_options,
            session_id=session_id,
        )

        assert "8675309" in output, (
            f"Expected '8675309' in resumed REPL output, got: {output!r}"
        )


class TestOneShotSessionResume:
    """One-shot ``--session`` loads prior context before running the turn."""

    def test_resume_preserves_context(
        self,
        local_llama_options: list[str],
        session_dir: Path,
    ) -> None:
        """A second one-shot call with the same session ID sees the first.

        This exercises ``_run_session_turn()``, which calls
        ``_load_session_history()`` then ``app.run_turn()``, and is the
        primary code path for resuming sessions programmatically.
        """
        session_id = f"oneshot-resume-{uuid.uuid4().hex[:8]}"

        # First call: plant information.
        _run_agent(
            "Remember that the secret word is kumquat. Reply with just 'ok'.",
            local_llama_options,
            session_id=session_id,
        )

        # Second call: retrieve it.
        result = _run_agent(
            "What is the secret word I told you? Answer with just one word.",
            local_llama_options,
            session_id=session_id,
        )

        assert "kumquat" in result.lower(), (
            f"Expected 'kumquat' in resumed answer, got: {result!r}"
        )


class TestSessionIsolation:
    """Sessions with different IDs produce independent contexts."""

    def test_independent_sessions_do_not_leak(
        self,
        local_llama_options: list[str],
        session_dir: Path,
    ) -> None:
        """Conversation written to session A is invisible to session B.

        This verifies the fundamental isolation contract: the session ID
        is the sole scoping key, and the repository/filename logic does
        not accidentally merge or cross-read events.
        """
        session_a = f"iso-a-{uuid.uuid4().hex[:8]}"
        session_b = f"iso-b-{uuid.uuid4().hex[:8]}"

        # ---- Plant different facts in A and B. ----
        _run_agent(
            "Remember that the secret word is plum. Reply with just 'ok'.",
            local_llama_options,
            session_id=session_a,
        )
        _run_agent(
            "Remember that the secret word is mango. Reply with just 'ok'.",
            local_llama_options,
            session_id=session_b,
        )

        # ---- Session A must report "plum", never "mango". ----
        result_a = _run_agent(
            "What is the secret word I told you? Answer with just one word.",
            local_llama_options,
            session_id=session_a,
        )
        assert "plum" in result_a.lower(), (
            f"Session A expected 'plum', got: {result_a!r}"
        )
        assert "mango" not in result_a.lower(), (
            f"Session A leaked content from session B: {result_a!r}"
        )

        # ---- Session B must report "mango", never "plum". ----
        result_b = _run_agent(
            "What is the secret word I told you? Answer with just one word.",
            local_llama_options,
            session_id=session_b,
        )
        assert "mango" in result_b.lower(), (
            f"Session B expected 'mango', got: {result_b!r}"
        )
        assert "plum" not in result_b.lower(), (
            f"Session B leaked content from session A: {result_b!r}"
        )
