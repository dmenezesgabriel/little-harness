"""Pure Python implementation of a ripgrep-like grep tool.

Isolates directory walking, file filtering, binary safety checking,
and regular expression searching entirely in standard Python.
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

KNOWN_TYPES: dict[str, set[str]] = {
    "py": {".py"},
    "md": {".md"},
    "txt": {".txt"},
    "json": {".json"},
    "yaml": {".yaml", ".yml"},
    "toml": {".toml"},
    "html": {".html", ".htm"},
    "css": {".css"},
    "js": {".js"},
    "ts": {".ts"},
}


@dataclass(frozen=True)
class RipgrepOutcome:
    """ripgrep's result. `exit_code` is None when search timed out or errored.

    By ripgrep's convention: 0 means matches, 1 means no matches, 2+ an error.
    """

    exit_code: int | None
    stdout: str
    stderr: str


class RipgrepSearch(Protocol):
    """Protocol defining the interface for grep search backends."""

    def run(
        self, arguments: Sequence[str], timeout_seconds: float, /
    ) -> RipgrepOutcome:
        """Run grep with the given arguments and capture its outcome."""
        ...


@dataclass(frozen=True)
class GrepRequest:
    """Typed request payload parsed from command-line arguments."""

    pattern: re.Pattern[str]
    paths: tuple[Path, ...]
    file_extensions: frozenset[str] | None
    max_results_per_file: int | None
    hidden: bool = False


class _GrepParseError(Exception):
    """Signaled when argument parsing encounters a specification error."""

    def __init__(self, outcome: RipgrepOutcome) -> None:
        super().__init__(outcome.stderr)
        self.outcome = outcome


def _string_list() -> list[str]:
    """Provide a type-safe empty list factory."""
    return []


@dataclass
class _ParsedArguments:
    """Intermediate storage during argument parsing."""

    ignore_case: bool = False
    file_type: str | None = None
    max_count_str: str | None = None
    hidden: bool = False
    positionals: list[str] = field(default_factory=_string_list)


class GrepArgumentParser:
    """Parses command line arguments for the Python grep tool."""

    def parse(self, arguments: Sequence[str]) -> GrepRequest | RipgrepOutcome:
        """Parse raw command line arguments into a structured GrepRequest.

        Example:
            request = parser.parse(["TODO", "."])

        """
        try:
            parsed_flags = self._parse_flags(arguments)
            if isinstance(parsed_flags, RipgrepOutcome):
                return parsed_flags
            return self._build_request(parsed_flags)
        except _GrepParseError as err:
            return err.outcome

    def _parse_flags(
        self, arguments: Sequence[str]
    ) -> _ParsedArguments | RipgrepOutcome:
        parsed = _ParsedArguments()
        i = 0
        n = len(arguments)
        while i < n:
            arg = arguments[i]
            if arg == "--":
                parsed.positionals.extend(arguments[i + 1 :])
                return parsed
            if arg.startswith("-") and arg != "-":
                outcome = self._handle_flag(arg, i, arguments, parsed)
                if isinstance(outcome, RipgrepOutcome):
                    return outcome
                i = outcome
                i += 1
                continue
            parsed.positionals.append(arg)
            i += 1
        return parsed

    def _handle_flag(
        self, flag: str, index: int, arguments: Sequence[str], parsed: _ParsedArguments
    ) -> int | RipgrepOutcome:
        if flag == "--hidden":
            parsed.hidden = True
            return index
        if flag in ("-i", "--ignore-case"):
            parsed.ignore_case = True
            return index

        res: int | RipgrepOutcome = RipgrepOutcome(
            exit_code=2,
            stdout="",
            stderr=f"Error: unknown flag {flag}",
        )
        if flag in ("-t", "--type"):
            res = self._handle_type_flag(index, arguments, parsed)
        if flag in ("-m", "--max-count"):
            res = self._handle_max_count_flag(index, arguments, parsed)
        return res

    def _handle_type_flag(
        self, index: int, arguments: Sequence[str], parsed: _ParsedArguments
    ) -> int | RipgrepOutcome:
        if index + 1 >= len(arguments):
            return RipgrepOutcome(
                exit_code=2,
                stdout="",
                stderr="Error: missing argument for -t/--type flag",
            )
        parsed.file_type = arguments[index + 1]
        return index + 1

    def _handle_max_count_flag(
        self, index: int, arguments: Sequence[str], parsed: _ParsedArguments
    ) -> int | RipgrepOutcome:
        if index + 1 >= len(arguments):
            return RipgrepOutcome(
                exit_code=2,
                stdout="",
                stderr="Error: missing argument for -m/--max-count flag",
            )
        parsed.max_count_str = arguments[index + 1]
        return index + 1

    def _build_request(self, parsed: _ParsedArguments) -> GrepRequest:
        if not parsed.positionals:
            raise _GrepParseError(
                RipgrepOutcome(2, "", "Error: missing search pattern")
            )
        pattern = self._compile_pattern(parsed.positionals[0], parsed.ignore_case)
        extensions = self._resolve_extensions(parsed.file_type)
        limit = self._resolve_max_results(parsed.max_count_str)
        paths = self._resolve_paths(parsed.positionals[1:])
        return GrepRequest(pattern, paths, extensions, limit, parsed.hidden)

    def _compile_pattern(self, pattern_str: str, ignore_case: bool) -> re.Pattern[str]:
        flags = re.IGNORECASE if ignore_case else 0
        try:
            return re.compile(pattern_str, flags)
        except re.error as err:
            raise _GrepParseError(
                RipgrepOutcome(
                    exit_code=2,
                    stdout="",
                    stderr=f"Error: invalid regex {pattern_str!r}: {err}",
                )
            ) from err

    def _resolve_extensions(self, file_type: str | None) -> frozenset[str] | None:
        if file_type is None:
            return None
        if file_type not in KNOWN_TYPES:
            raise _GrepParseError(
                RipgrepOutcome(
                    exit_code=2,
                    stdout="",
                    stderr=f"Error: unknown type {file_type!r}",
                )
            )
        return frozenset(KNOWN_TYPES[file_type])

    def _resolve_max_results(self, max_count_str: str | None) -> int | None:
        if max_count_str is None:
            return None
        try:
            return int(max_count_str)
        except ValueError as err:
            raise _GrepParseError(
                RipgrepOutcome(
                    exit_code=2,
                    stdout="",
                    stderr=f"Error: invalid max-count value {max_count_str!r}",
                )
            ) from err

    def _resolve_paths(self, raw_paths: list[str]) -> tuple[Path, ...]:
        if not raw_paths:
            return (Path(),)
        paths: list[Path] = []
        for p_str in raw_paths:
            p = Path(p_str)
            if not p.exists():
                raise _GrepParseError(
                    RipgrepOutcome(
                        exit_code=2,
                        stdout="",
                        stderr=f"Error: path does not exist: {p_str}",
                    )
                )
            paths.append(p)
        return tuple(paths)


class PythonGrepSearch:
    """Pure Python implementation of grep search."""

    def __init__(self, parser: GrepArgumentParser | None = None) -> None:
        """Initialize the PythonGrepSearch with an optional argument parser.

        Example:
            search = PythonGrepSearch()

        """
        self._parser = parser or GrepArgumentParser()

    def run(self, arguments: Sequence[str], timeout_seconds: float) -> RipgrepOutcome:
        """Execute a grep search over the local filesystem.

        Example:
            outcome = search.run(["TODO", "src"], 30.0)

        """
        request = self._parser.parse(arguments)
        if isinstance(request, RipgrepOutcome):
            return request
        return self._execute_search(request, timeout_seconds)

    def _execute_search(
        self, request: GrepRequest, timeout_seconds: float
    ) -> RipgrepOutcome:
        deadline = time.monotonic() + timeout_seconds
        all_matches: list[str] = []
        for path in request.paths:
            for file_path in self._walk_files(path, request.hidden):
                if time.monotonic() > deadline:
                    break
                if self._should_skip(file_path, request.file_extensions):
                    continue
                matches = self._grep_file(
                    file_path,
                    request.pattern,
                    request.max_results_per_file,
                )
                all_matches.extend(matches)
            if time.monotonic() > deadline:
                break
        return self._format_results(all_matches)

    def _walk_files(self, path: Path, hidden: bool | None = None) -> Iterator[Path]:
        is_hidden = False if hidden is None else bool(hidden)
        if path.is_file():
            yield path
            return
        for root, dirs, files in os.walk(path):
            dirs[:] = [
                d
                for d in dirs
                if (is_hidden or not d.startswith(".")) and d != "__pycache__"
            ]
            for file in files:
                if not is_hidden and file.startswith("."):
                    continue
                yield Path(root) / file

    def _should_skip(
        self, file_path: Path, file_extensions: frozenset[str] | None
    ) -> bool:
        if (
            file_extensions is not None
            and file_path.suffix.lower() not in file_extensions
        ):
            return True
        return self._is_binary(file_path)

    def _is_binary(self, file_path: Path) -> bool:
        try:
            with file_path.open("rb") as f:
                return b"\x00" in f.read(1024)
        except OSError:
            return True

    def _grep_file(
        self,
        file_path: Path,
        pattern: re.Pattern[str],
        max_results: int | None,
    ) -> list[str]:
        matches: list[str] = []
        try:
            with file_path.open("r", encoding="utf-8", errors="replace") as f:
                for line_num, line in enumerate(f, start=1):
                    line_stripped = line.rstrip("\r\n")
                    if pattern.search(line_stripped):
                        matches.append(f"{file_path}:{line_num}:{line_stripped}")
                        if max_results is not None and len(matches) >= max_results:
                            break
        except OSError:
            pass
        return matches

    def _format_results(self, all_matches: list[str]) -> RipgrepOutcome:
        if not all_matches:
            return RipgrepOutcome(exit_code=1, stdout="", stderr="")
        stdout_content = "\n".join(all_matches) + "\n"
        return RipgrepOutcome(exit_code=0, stdout=stdout_content, stderr="")
