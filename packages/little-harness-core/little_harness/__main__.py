"""Console-script entry point: `little-harness`.

Example:
    little-harness --provider llama_cpp -o model_path=models/m.gguf -p "2 + 2?"
"""

from __future__ import annotations

from little_harness.composition import run_cli


def main() -> None:
    print(run_cli())


if __name__ == "__main__":
    main()
