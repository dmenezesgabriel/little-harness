#!/usr/bin/env python3
"""Aggregate benchmark JSONL into the markdown tables the report needs.

Reads one or more ``*.jsonl`` files produced by ``benchmark.py`` and prints
per-model and per-tool-set rollups: success/answer rates, latency decomposition
(TTFT, generation throughput, end-to-end), tool-call usage, and the
hallucination rate (trials where the model answered without calling any tool).

Usage
-----
    python -m evaluation.summarize results/sweep_models.jsonl
    python -m evaluation.summarize results/*.jsonl --group model,tool_set
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

Trial = dict[str, object]


def load_trials(paths: Sequence[Path]) -> list[Trial]:
    """Read every JSONL line across `paths` into a flat list of trial dicts."""
    trials: list[Trial] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                trials.append(json.loads(line))
    return trials


def _mean(values: Sequence[float]) -> float:
    """Mean of `values`, or 0.0 when empty."""
    return round(statistics.mean(values), 2) if values else 0.0


def _rate(flags: Sequence[bool]) -> str:
    """Format the fraction of true flags as a percentage."""
    if not flags:
        return "-"
    return f"{100 * sum(flags) / len(flags):.0f}%"


@dataclass(frozen=True)
class Rollup:
    """Aggregate stats for one group of trials."""

    key: str
    count: int
    pass_rate: str
    answer_rate: str
    hallucination_rate: str
    ttft: float
    gen_tps: float
    end_to_end: float
    tool_calls: float


def summarize_group(key: str, trials: Sequence[Trial]) -> Rollup:
    """Roll up one group of trials into reportable aggregates."""
    ttfts = [t["first_token_seconds"] for t in trials if t["first_token_seconds"]]
    return Rollup(
        key=key,
        count=len(trials),
        pass_rate=_rate([bool(t["succeeded"]) for t in trials]),
        answer_rate=_rate([bool(t["answer_correct"]) for t in trials]),
        hallucination_rate=_rate([t["tool_call_count"] == 0 for t in trials]),
        ttft=_mean([float(x) for x in ttfts]),
        gen_tps=_mean([float(t["generation_tokens_per_second"]) for t in trials]),
        end_to_end=_mean([float(t["end_to_end_seconds"]) for t in trials]),
        tool_calls=_mean([float(t["tool_call_count"]) for t in trials]),
    )


def group_by(trials: Sequence[Trial], key_of: Callable[[Trial], str]) -> list[Rollup]:
    """Group trials by `key_of` and roll each group up, sorted by key."""
    buckets: dict[str, list[Trial]] = {}
    for trial in trials:
        buckets.setdefault(key_of(trial), []).append(trial)
    return [summarize_group(key, group) for key, group in sorted(buckets.items())]


def render_table(rollups: Sequence[Rollup], key_header: str) -> str:
    """Render rollups as a GitHub-flavored markdown table."""
    head = (
        f"| {key_header} | n | pass | answer | halluc | TTFT s | gen tok/s | "
        "e2e s | tool calls |\n"
        "| --- | --: | --: | --: | --: | --: | --: | --: | --: |"
    )
    rows = [
        f"| {r.key} | {r.count} | {r.pass_rate} | {r.answer_rate} | "
        f"{r.hallucination_rate} | {r.ttft} | {r.gen_tps} | {r.end_to_end} | "
        f"{r.tool_calls} |"
        for r in rollups
    ]
    return "\n".join([head, *rows])


def _key_of(field_names: Sequence[str]) -> Callable[[Trial], str]:
    """Build a grouping key function joining the given trial fields."""
    return lambda trial: " / ".join(str(trial[name]) for name in field_names)


def main() -> None:
    """Parse args, load trials, and print one markdown table per grouping."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument(
        "--group",
        default="model",
        help="comma-separated trial fields to group by (e.g. model,tool_set)",
    )
    args = parser.parse_args()
    trials = load_trials(args.paths)
    fields = args.group.split(",")
    rollups = group_by(trials, _key_of(fields))
    print(f"\n## Grouped by {' / '.join(fields)}  ({len(trials)} trials)\n")
    print(render_table(rollups, args.group))


if __name__ == "__main__":
    main()
