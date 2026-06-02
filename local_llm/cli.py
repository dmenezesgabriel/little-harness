from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from local_llm.agent import AgentResult, AgentStep

MODEL_PATH: Final[Path] = Path("models/LFM2-8B-A1B-Q4_K_M.gguf")


@dataclass(frozen=True)
class AppConfig:
    prompt: str
    model_path: Path
    context_size: int
    thread_count: int
    gpu_layer_count: int
    temperature: float
    max_tokens: int
    max_iterations: int


def parse_args() -> AppConfig:
    parser = argparse.ArgumentParser(description="Run a small local LLM agent.")
    add_prompt_args(parser)
    add_model_args(parser)
    add_runtime_args(parser)

    namespace = parser.parse_args()

    return AppConfig(
        prompt=str(namespace.prompt),
        model_path=Path(str(namespace.model_path)),
        context_size=int(namespace.ctx),
        thread_count=int(namespace.threads),
        gpu_layer_count=int(namespace.gpu_layers),
        temperature=float(namespace.temperature),
        max_tokens=int(namespace.max_tokens),
        max_iterations=int(namespace.max_iterations),
    )


def add_prompt_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-p",
        "--prompt",
        default=(
            "Explain llama.cpp in exactly 3 short bullet points. "
            "Be specific: mention GGUF models, local inference, "
            "and CPU-friendly execution."
        ),
        help="Prompt to send to the local model.",
    )


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model-path",
        default=str(MODEL_PATH),
        help="Path to the local GGUF model.",
    )
    parser.add_argument("--ctx", type=int, default=8192, help="Context size.")
    parser.add_argument("--threads", type=int, default=8, help="CPU thread count.")
    parser.add_argument(
        "--gpu-layers",
        type=int,
        default=0,
        help="Number of GPU layers. Use 0 for CPU-only.",
    )


def add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum generated tokens.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Maximum agent loop iterations.",
    )


def print_result(result: AgentResult) -> None:
    print(result.answer)
    print(f"\nElapsed: {result.elapsed_seconds:.2f}s")

    if len(result.steps) == 0:
        return

    print("\nAgent steps:")

    for step in result.steps:
        print(f"\nStep {step.iteration}")
        print(f"Action: {format_step_action(step)}")
        print(f"Observation: {step.observation}")


def format_step_action(step: AgentStep) -> str:
    if step.decision is None:
        return "repair"

    if step.decision.kind == "final":
        return "final"

    return step.decision.tool_name or "tool"
