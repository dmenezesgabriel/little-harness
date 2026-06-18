#!/usr/bin/env python3
"""Model-reusing latency/quality benchmark for small local LLMs.

The existing ``run.py`` harness reloads the GGUF for every trial (it shells
through ``run_cli``); on CPU with 4-5 GB models the reload dominates wall time.
This engine loads each model **once** and drives many trials in-process,
reusing the warm model across runtime/tool variations. It captures the new
``on_model_metrics`` telemetry (TTFT, generated tokens, throughput) plus
tool-call and repair counts, and scores each trial with the false-positive
guards the old harness lacked (``forbidden_substring`` and ``min_tool_calls``).

Usage
-----
    python -m evaluation.benchmark --model models/LFM2.5-1.2B-Instruct-Q8_0.gguf \
        --threads 4 --batch 256 --no-flash-attn --out results/run.jsonl

One process = one model load (+ one set of load-time options: threads, batch,
flash-attn, n_ctx). Runtime options (temperature, max-iterations) and tool sets
are swept in-process. A shell driver launches one process per (model, load-opts).
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TextIO

from little_harness.application.agent_dependencies import AgentDependencies
from little_harness.application.agent_runtime import AgentRuntime, AgentRuntimeConfig
from little_harness.application.hook_chain import HookChain
from little_harness.application.ports.chat_model import ChatModel
from little_harness.application.tool_registry import ToolRegistry
from little_harness.composition import approval_required_names
from little_harness.domain.tool_result import ToolRunResult
from little_harness.domain.values.model_call_metrics import ModelCallMetrics
from little_harness.domain.values.numeric_values import (
    ElapsedSeconds,
    Iteration,
    MaxIterations,
    MaxTokens,
    Temperature,
)
from little_harness.domain.values.text_values import Prompt, RunId
from little_harness.domain.values.truncation import TruncationConfig
from little_harness.infrastructure.hooks.approval_hook import ApprovalHook
from little_harness.infrastructure.observability.null_observer import NullObserver
from little_harness.infrastructure.permissions.auto_approve_requester import (
    AutoApprovePermissionRequester,
)
from little_harness.infrastructure.skills.file_system_skill_loader import (
    FileSystemSkillLoader,
)
from little_harness.infrastructure.truncation.head_truncator import HeadTruncator
from little_harness.plugin_discovery import (
    discover_policy,
    discover_tools,
    load_chat_model_builder,
)
from little_harness.presentation.cli.token_sinks import NullTokenSink

from evaluation.suite import EvalCase, ToolSet, load_suite, select_tool_sets


class MetricsObserver(NullObserver):
    """Captures per-call latency metrics, tool calls, and repairs for one trial."""

    def __init__(self) -> None:
        """Start with empty metric, tool-call, and repair accumulators."""
        self.metrics: list[ModelCallMetrics] = []
        self.tool_calls: list[tuple[str, bool]] = []
        self.repairs = 0

    def on_model_metrics(
        self, run_id: RunId, iteration: Iteration, metrics: ModelCallMetrics
    ) -> None:
        """Record one model call's latency metrics."""
        del run_id, iteration
        self.metrics.append(metrics)

    def on_tool_invoked(
        self,
        run_id: RunId,
        iteration: Iteration,
        result: ToolRunResult,
        elapsed: ElapsedSeconds,
    ) -> None:
        """Record one tool call's name and success."""
        del run_id, iteration, elapsed
        self.tool_calls.append((result.tool_name.value, result.succeeded))

    def on_repair(self, run_id: RunId, iteration: Iteration, error: Exception) -> None:
        """Count one protocol-repair attempt."""
        del run_id, iteration, error
        self.repairs += 1


@dataclass
class TrialRecord:
    """One scored trial with its full configuration and latency decomposition."""

    model: str
    threads: int
    batch: int
    flash_attn: bool
    n_ctx: int
    temperature: float
    max_iterations: int
    tool_set: str
    tools: list[str]
    case_id: str
    succeeded: bool
    answer_correct: bool
    end_to_end_seconds: float
    first_token_seconds: float | None
    total_output_tokens: int
    generation_tokens_per_second: float
    model_calls: int
    tool_call_count: int
    tool_names: list[str]
    repair_count: int
    iterations_used: int
    output_excerpt: str
    error: str | None = None
    failure_reasons: list[str] = field(default_factory=list)


def build_reused_dependencies(
    chat_model: ChatModel, tool_names: Sequence[str], observer: MetricsObserver
) -> AgentDependencies:
    """Assemble dependencies that reuse the given model, auto-approving tools."""
    registry = ToolRegistry(discover_tools(tuple(tool_names)))
    approval = approval_required_names(registry)
    hooks: list[ApprovalHook] = []
    if approval:
        hooks.append(ApprovalHook(AutoApprovePermissionRequester(), approval))
    return AgentDependencies(
        chat_model=chat_model,
        tool_registry=registry,
        policy=discover_policy("json"),
        observer=observer,
        token_sink=NullTokenSink(),
        hooks=HookChain(list(hooks)),
        truncator=HeadTruncator(),
        truncation_config=TruncationConfig(),
        skill_loader=FileSystemSkillLoader(()),
    )


def generation_throughput(metrics: Sequence[ModelCallMetrics]) -> float:
    """Tokens per second of pure generation, excluding prefill/TTFT.

    Isolating generation from time-to-first-token matters on CPU, where prefill
    of a large system prompt (and GBNF grammar compile) can dwarf decoding.
    """
    tokens = 0
    seconds = 0.0
    for metric in metrics:
        if metric.time_to_first_token is None:
            continue
        tokens += metric.output_tokens
        seconds += metric.elapsed.value - metric.time_to_first_token.value
    if seconds <= 0:
        return 0.0
    return tokens / seconds


def answer_is_correct(
    output: str, expected_substring: str, forbidden_substrings: Sequence[str]
) -> bool:
    """True when the expected text is present and no forbidden text leaked.

    Matching is separator-insensitive: small models format numbers with
    thousands commas (``1,024``), so a literal ``1024`` must still match.
    """
    normalized = output.replace(",", "")
    if expected_substring and expected_substring not in normalized:
        return False
    return not any(bad.lower() in output.lower() for bad in forbidden_substrings)


def score_trial(
    output: str,
    observer: MetricsObserver,
    expected_substring: str,
    forbidden_substrings: Sequence[str],
    min_tool_calls: int,
) -> list[str]:
    """Return the list of failure reasons; empty means the trial passed."""
    reasons: list[str] = []
    if not answer_is_correct(output, expected_substring, forbidden_substrings):
        reasons.append(f"answer wrong/forbidden (expected {expected_substring!r})")
    if len(observer.tool_calls) < min_tool_calls:
        reasons.append(
            f"only {len(observer.tool_calls)} tool calls, expected >= {min_tool_calls}"
        )
    return reasons


@dataclass(frozen=True)
class RuntimeSpec:
    """The in-process (no-reload) trial variables: sampling, loop cap, tool set."""

    temperature: float
    max_iterations: int
    tool_set_name: str
    tools: tuple[str, ...]


def run_trial(
    chat_model: ChatModel,
    load: LoadConfig,
    spec: RuntimeSpec,
    case: EvalCase,
) -> TrialRecord:
    """Run one prompt through a freshly wired (model-reusing) runtime and score it."""
    observer = MetricsObserver()
    dependencies = build_reused_dependencies(chat_model, spec.tools, observer)
    runtime = AgentRuntime(
        dependencies,
        AgentRuntimeConfig(
            max_iterations=MaxIterations(spec.max_iterations),
            temperature=Temperature(spec.temperature),
            max_tokens=MaxTokens(512),
        ),
    )
    start = time.perf_counter()
    error: str | None = None
    output = ""
    try:
        output = runtime.run(Prompt(case.prompt)).answer.value
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    end_to_end = time.perf_counter() - start

    reasons = (
        [f"exception: {error}"]
        if error is not None
        else score_trial(
            output,
            observer,
            case.expected_substring,
            case.forbidden_substrings,
            case.min_tool_calls,
        )
    )
    first = observer.metrics[0].time_to_first_token if observer.metrics else None
    return TrialRecord(
        model=Path(load.model_path).name,
        threads=load.threads,
        batch=load.batch,
        flash_attn=load.flash_attn,
        n_ctx=load.n_ctx,
        temperature=spec.temperature,
        max_iterations=spec.max_iterations,
        tool_set=spec.tool_set_name,
        tools=list(spec.tools),
        case_id=case.case_id,
        succeeded=error is None and not reasons,
        answer_correct=error is None
        and answer_is_correct(
            output, case.expected_substring, case.forbidden_substrings
        ),
        end_to_end_seconds=round(end_to_end, 3),
        first_token_seconds=None if first is None else round(first.value, 3),
        total_output_tokens=sum(m.output_tokens for m in observer.metrics),
        generation_tokens_per_second=round(generation_throughput(observer.metrics), 2),
        model_calls=len(observer.metrics),
        tool_call_count=len(observer.tool_calls),
        tool_names=[name for name, _ok in observer.tool_calls],
        repair_count=observer.repairs,
        iterations_used=len(observer.metrics),
        output_excerpt=output[:200].replace("\n", " "),
        error=error,
        failure_reasons=reasons,
    )


@dataclass(frozen=True)
class LoadConfig:
    """Model load-time options (a change here forces a model reload)."""

    model_path: str
    threads: int
    batch: int
    flash_attn: bool
    n_ctx: int

    def provider_options(self) -> dict[str, str]:
        """Render as the llama_cpp provider option mapping."""
        return {
            "model_path": self.model_path,
            "n_threads": str(self.threads),
            "n_batch": str(self.batch),
            "flash_attn": "true" if self.flash_attn else "false",
            "n_ctx": str(self.n_ctx),
            "n_gpu_layers": "0",
            "seed": "42",
        }


def load_model(load: LoadConfig) -> ChatModel:
    """Load the llama.cpp model once for the whole process."""
    builder = load_chat_model_builder("llama_cpp")
    return builder(load.provider_options())


def run_matrix(
    chat_model: ChatModel,
    load: LoadConfig,
    temperatures: Sequence[float],
    iteration_caps: Sequence[int],
    tool_sets: Sequence[ToolSet],
    out_handle: TextIO,
) -> int:
    """Run the full in-process matrix, streaming each trial as a JSONL line."""
    suite = load_suite()
    count = 0
    for temperature in temperatures:
        for max_iterations in iteration_caps:
            for tool_set in tool_sets:
                for case in suite:
                    if not tool_set.runnable(case):
                        continue
                    spec = RuntimeSpec(
                        temperature=temperature,
                        max_iterations=max_iterations,
                        tool_set_name=tool_set.name,
                        tools=tool_set.tools_for(case),
                    )
                    record = run_trial(chat_model, load, spec, case)
                    out_handle.write(json.dumps(asdict(record)) + "\n")
                    out_handle.flush()
                    status = "PASS" if record.succeeded else "FAIL"
                    print(
                        f"  [{tool_set.name:>10}] {case.case_id:28s} t={temperature} "
                        f"it={max_iterations} {status} "
                        f"{record.end_to_end_seconds:6.1f}s "
                        f"ttft={record.first_token_seconds} "
                        f"gen={record.generation_tokens_per_second}tok/s "
                        f"tools={record.tool_call_count}"
                    )
                    count += 1
    return count


def parse_args() -> argparse.Namespace:
    """Parse the benchmark CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="GGUF path")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--flash-attn", action="store_true", default=False)
    parser.add_argument("--temperatures", default="0.1")
    parser.add_argument("--max-iterations", default="5")
    parser.add_argument(
        "--tool-sets", default="single,curated,all", help="suite tool-set names"
    )
    parser.add_argument("--out", required=True, help="JSONL output path (appended)")
    return parser.parse_args()


def _drive(load: LoadConfig, args: argparse.Namespace) -> None:
    """Load the model once and run the full configured matrix to JSONL."""
    temperatures = [float(t) for t in args.temperatures.split(",")]
    iteration_caps = [int(i) for i in args.max_iterations.split(",")]
    tool_sets = select_tool_sets(args.tool_sets.split(","))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"Loading {Path(load.model_path).name} (threads={load.threads}, "
        f"batch={load.batch}, flash_attn={load.flash_attn}, n_ctx={load.n_ctx}) ..."
    )
    started = time.perf_counter()
    chat_model = load_model(load)
    print(f"  loaded in {time.perf_counter() - started:.1f}s")
    try:
        with out_path.open("a", encoding="utf-8") as handle:
            total = run_matrix(
                chat_model, load, temperatures, iteration_caps, tool_sets, handle
            )
        print(f"Wrote {total} trials to {out_path}")
    finally:
        chat_model.close()


def main() -> None:
    """Parse args, run the matrix, and report peak RSS for the bloat audit."""
    args = parse_args()
    load = LoadConfig(
        model_path=args.model,
        threads=args.threads,
        batch=args.batch,
        flash_attn=args.flash_attn,
        n_ctx=args.n_ctx,
    )
    _drive(load, args)
    # ru_maxrss is KiB on Linux; report peak resident set for the bloat audit.
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"PEAK_RSS_MB {Path(load.model_path).name} {peak_mb:.0f}")


if __name__ == "__main__":
    main()
