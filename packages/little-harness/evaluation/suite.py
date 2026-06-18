"""Production-realistic evaluation cases and tool-set definitions.

Cases operate on real repository files (no setup, non-destructive) so the
benchmark measures tool-calling against content a model would actually face.
Each case names the tools it genuinely needs; the three tool sets then vary how
many *extra* tools surround them, isolating the prompt-size/selection latency
tax from task difficulty.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

# Every installed tool — the "all" set the model must choose from.
ALL_TOOLS: tuple[str, ...] = (
    "calculator",
    "read_file",
    "write_file",
    "edit_file",
    "bash",
    "find",
    "ls",
    "ripgrep",
    "web_fetch",
    "ast_grep",
    "ast_edit",
)

# The proven-reliable subset for small models (calculator/read_file/edit_file/
# ripgrep/ls), per the 2026-06-14 report: excludes ast_* (unusable) and the
# bash/web_fetch escape hatches that invite hallucination.
CURATED_TOOLS: tuple[str, ...] = (
    "calculator",
    "read_file",
    "edit_file",
    "ripgrep",
    "ls",
)


@dataclass(frozen=True)
class EvalCase:
    """One scored prompt: what to ask, what proves success, what it needs."""

    case_id: str
    category: str
    prompt: str
    expected_substring: str
    required_tools: tuple[str, ...]
    min_tool_calls: int = 1
    forbidden_substrings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ToolSet:
    """A named strategy for choosing which tools a case is offered."""

    name: str

    def tools_for(self, case: EvalCase) -> tuple[str, ...]:
        """Resolve the tool list this set offers for the given case."""
        if self.name == "single":
            return case.required_tools
        if self.name == "curated":
            return CURATED_TOOLS
        return ALL_TOOLS

    def runnable(self, case: EvalCase) -> bool:
        """True when every tool the case needs is present in this set."""
        offered = set(self.tools_for(case))
        return set(case.required_tools).issubset(offered)


_TOOL_SETS = {name: ToolSet(name) for name in ("single", "curated", "all")}


def select_tool_sets(names: Sequence[str]) -> list[ToolSet]:
    """Resolve tool-set names, raising on an unknown one."""
    result: list[ToolSet] = []
    for name in names:
        key = name.strip()
        if key not in _TOOL_SETS:
            raise ValueError(
                f"Unknown tool set: {key!r}. Expected one of {sorted(_TOOL_SETS)}."
            )
        result.append(_TOOL_SETS[key])
    return result


_NOT_FOUND = ("No such file", "No matches", "not found")


def load_suite() -> list[EvalCase]:
    """Return the fixed evaluation suite (deterministic against repo files)."""
    return [
        EvalCase(
            "arith_division",
            "arithmetic",
            "Use the calculator tool to work out 144 divided by 12, then state the "
            "numeric result.",
            "12",
            ("calculator",),
        ),
        EvalCase(
            "arith_power",
            "arithmetic",
            "Use the calculator tool to compute 2 raised to the 10th power and "
            "report the exact integer.",
            "1024",
            ("calculator",),
        ),
        EvalCase(
            "arith_compound",
            "arithmetic",
            "Use the calculator tool to evaluate (10 + 5) * 3 and report the result.",
            "45",
            ("calculator",),
        ),
        EvalCase(
            "arith_modulo",
            "arithmetic",
            "Use the calculator tool to compute 100 modulo 7 and report the remainder.",
            "2",
            ("calculator",),
        ),
        EvalCase(
            "read_line_length",
            "read",
            "Read the file pyproject.toml and report the exact integer configured "
            "for ruff's line-length setting.",
            "88",
            ("read_file",),
            forbidden_substrings=_NOT_FOUND,
        ),
        EvalCase(
            "read_python_version",
            "read",
            "Read the file .python-version and report the exact version string it "
            "contains.",
            "3.12",
            ("read_file",),
            forbidden_substrings=_NOT_FOUND,
        ),
        EvalCase(
            "search_measure_stream",
            "search",
            "Search the packages directory for where the function named "
            "measure_stream is defined, and name the Python file that defines it.",
            "stream_timing",
            ("ripgrep",),
            forbidden_substrings=_NOT_FOUND,
        ),
        EvalCase(
            "search_fallback_answer",
            "search",
            "Search the packages directory for the identifier FALLBACK_ANSWER and "
            "name the Python file where it is assigned.",
            "agent_runtime",
            ("ripgrep",),
            forbidden_substrings=_NOT_FOUND,
        ),
        EvalCase(
            "list_models_gguf",
            "list",
            "List the contents of the models directory and report how many files "
            "have the .gguf extension.",
            "6",
            ("ls",),
            forbidden_substrings=_NOT_FOUND,
        ),
        EvalCase(
            "list_core_package",
            "list",
            "List the packages directory and report the full name of the directory "
            "that ends with 'core'.",
            "little-harness-core",
            ("ls",),
            forbidden_substrings=_NOT_FOUND,
        ),
        EvalCase(
            "multi_read_then_calc",
            "multi_step",
            "Read pyproject.toml to find ruff's line-length value, then use the "
            "calculator tool to multiply that value by 2 and report the product.",
            "176",
            ("read_file", "calculator"),
            min_tool_calls=2,
            forbidden_substrings=_NOT_FOUND,
        ),
    ]


def runnable_cases(tool_set: ToolSet) -> list[EvalCase]:
    """Return the suite cases runnable under `tool_set`."""
    return [case for case in load_suite() if tool_set.runnable(case)]
