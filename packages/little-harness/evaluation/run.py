#!/usr/bin/env python3
"""Orchestrate the tool-overlap evaluation suite.

Usage
-----
    python -m evaluation.run                          # llama.cpp only (default)
    python -m evaluation.run --provider all           # both providers
    python -m evaluation.run --provider litellm       # remote Gemini only
    python -m evaluation.run --tasks read_file,arithmetic  # specific task files
    python -m evaluation.run --output results.json    # custom output path

Each task YAML in ``tasks/`` defines a set of evaluation cases and the
candidate tools to compare.  Every case is run once per candidate tool and
the outcomes are aggregated into a JSON report.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from evaluation.runner import (
    TrialResult,
    resolve_litellm_options,
    resolve_llama_options,
    run_trial,
)


@dataclass
class TaskCase:
    """A single evaluation case loaded from YAML."""

    id: str
    setup: dict[str, str]
    prompt: str
    expected_substring: str
    candidates: list[str]


@dataclass
class TaskFile:
    """A YAML task file with its source path."""

    path: Path
    cases: list[TaskCase]


def load_task_files(
    task_dir: Path, filter_names: Sequence[str] | None
) -> list[TaskFile]:
    """Load task YAMLs from ``task_dir``, optionally filtered by stem name."""
    result: list[TaskFile] = []
    for path in sorted(task_dir.glob("*.yaml")):
        if filter_names and path.stem not in filter_names:
            continue
        raw = yaml.safe_load(path.read_text())
        cases = [TaskCase(**c) for c in raw["cases"]]
        result.append(TaskFile(path=path, cases=cases))
    return result


def run_setup(cases: list[TaskCase], workdir: Path) -> None:
    """Create files needed by the cases in ``workdir``."""
    for case in cases:
        for name, content in case.setup.items():
            target = workdir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")


def run_evaluation(
    provider: str,
    provider_options: list[str],
    task_files: list[TaskFile],
    workdir: Path,
) -> list[TrialResult]:
    """Run every case through every candidate tool; return all trial results."""
    results: list[TrialResult] = []
    for task_file in task_files:
        run_setup(task_file.cases, workdir)
        for case in task_file.cases:
            for tool in case.candidates:
                tools_list = [tool]
                result = run_trial(
                    task_id=f"{task_file.path.stem}/{case.id}",
                    prompt=case.prompt,
                    provider=provider,
                    provider_options=provider_options,
                    tools=tools_list,
                    expected_substring=case.expected_substring,
                )
                print(
                    f"  [{result.tool_name:>12}] {result.task_id:40s} "
                    f"{'PASS' if result.succeeded else 'FAIL'} "
                    f"({result.duration_seconds:.1f}s)"
                )
                results.append(result)
    return results


_PROVIDERS = {
    "llama_cpp": resolve_llama_options,
    "litellm": resolve_litellm_options,
}


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the evaluation runner."""
    parser = argparse.ArgumentParser(
        description="Run tool-overlap evaluations against real models."
    )
    parser.add_argument(
        "--provider",
        default="llama_cpp",
        choices=[*list(_PROVIDERS), "all"],
        help="Which provider to test (default: llama_cpp)",
    )
    parser.add_argument(
        "--tasks",
        default=None,
        help="Comma-separated task file stems (e.g. read_file,arithmetic). "
        "Default: all.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path for the JSON results file (default: evaluation_report.json)",
    )
    return parser


def resolve_providers(selected: str) -> list[tuple[str, list[str]]]:
    """Return (provider_name, options) pairs matching ``selected``."""
    result: list[tuple[str, list[str]]] = []
    for provider_name, resolver in _PROVIDERS.items():
        if selected not in ("all", provider_name):
            continue
        opts = resolver()
        if opts is None:
            print(
                f"[skip] {provider_name} prerequisites not satisfied", file=sys.stderr
            )
            continue
        result.append((provider_name, opts))
    return result


def print_summary(summary: list[dict]) -> None:
    """Print the aggregated summary table."""
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    for row in summary:
        status = "PASS" if row["pass_rate"] == 1.0 else f"{row['pass_rate']:.0%}"
        avg = row["avg_time"]
        print(
            f"  {row['tool']:>12}  {row['count']:3d} trials  {status}  ({avg:.1f}s avg)"
        )


def run_providers(
    providers: list[tuple[str, list[str]]], task_files: list[TaskFile]
) -> list[TrialResult]:
    """Run all providers through every task file."""
    results: list[TrialResult] = []
    for provider_name, provider_options in providers:
        print(f"\n{'=' * 60}")
        print(f"Provider: {provider_name}")
        print(f"{'=' * 60}")
        results.extend(
            run_evaluation(
                provider=provider_name,
                provider_options=provider_options,
                task_files=task_files,
                workdir=Path.cwd(),
            )
        )
    return results


def write_report(
    all_results: list[TrialResult], summary: list[dict], output_path: Path
) -> None:
    """Write the JSON report to ``output_path``."""
    report = {
        "providers": list({r.provider for r in all_results}),
        "summary": summary,
        "trials": [asdict(r) for r in all_results],
    }
    output_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport written to {output_path}")


def main() -> None:
    """Entry point: parse args, load tasks, run trials, write report."""
    args = build_parser().parse_args()
    task_dir = Path(__file__).resolve().parent / "tasks"
    filter_names = args.tasks.split(",") if args.tasks else None
    output_path = Path(args.output) if args.output else Path("evaluation_report.json")

    task_files = load_task_files(task_dir, filter_names)
    if not task_files:
        print(f"No task files found in {task_dir}", file=sys.stderr)
        sys.exit(1)

    providers = resolve_providers(args.provider)
    if not providers:
        print(
            "No providers available. Check model files and API keys.", file=sys.stderr
        )
        sys.exit(1)

    all_results = run_providers(providers, task_files)
    summary = aggregate(results=all_results)
    print_summary(summary)
    write_report(all_results, summary, output_path)


@dataclass
class ToolAggregate:
    """Per-tool aggregate counts for the summary report."""

    tool: str
    count: int
    passed: int
    total_time: float


def aggregate(results: list[TrialResult]) -> list[dict]:
    """Aggregate results per tool."""
    aggregator: dict[str, ToolAggregate] = {}
    for r in results:
        if r.tool_name not in aggregator:
            aggregator[r.tool_name] = ToolAggregate(
                tool=r.tool_name, count=0, passed=0, total_time=0.0
            )
        agg = aggregator[r.tool_name]
        agg.count += 1
        agg.total_time += r.duration_seconds
        if r.succeeded:
            agg.passed += 1
    return [
        {
            "tool": agg.tool,
            "count": agg.count,
            "pass_rate": agg.passed / agg.count if agg.count else 0.0,
            "avg_time": agg.total_time / agg.count if agg.count else 0.0,
        }
        for agg in sorted(aggregator.values(), key=lambda x: x.tool)
    ]


if __name__ == "__main__":
    main()
