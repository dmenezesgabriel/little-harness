"""Console-script entry point: `little-harness`.

Turns the agent run into a CLI: prints the answer, or a single concise
`error: …` line on stderr (with exit code 1) when a provider/tool or
configuration fails. Pass `--log` to re-raise and see the full traceback.

Example:
    little-harness --provider llama_cpp -o model_path=models/m.gguf -p "2 + 2?"
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from little_harness.composition import run_cli

TRACEBACK_FLAG = "--log"


def main() -> None:
    raise SystemExit(run(sys.argv[1:]))


def run(argv: Sequence[str]) -> int:
    try:
        print(run_cli(argv))
        return 0
    except Exception as error:  # top-level CLI boundary: errors become exit codes
        if TRACEBACK_FLAG in argv:
            raise

        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    main()
