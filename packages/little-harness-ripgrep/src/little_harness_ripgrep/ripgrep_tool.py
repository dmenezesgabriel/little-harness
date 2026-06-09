"""Tool that runs ripgrep over the workspace and returns its matches."""

from __future__ import annotations

import shlex

from little_harness.domain.tool_result import ToolRunRequest, ToolRunResult
from little_harness.domain.tool_spec import ToolExamples, ToolInputSchema, ToolSpec
from little_harness.domain.values.text_values import ToolName, ToolOutput

from little_harness_ripgrep.ripgrep_search import RipgrepOutcome, RipgrepSearch

DEFAULT_TIMEOUT_SECONDS = 30.0
NO_MATCH_EXIT_CODE = 1


class RipgrepTool:
    """Searches files with ripgrep; reading is safe, so no approval is needed.

    The raw input is a ripgrep argument line (pattern plus optional paths and
    flags), split with shell rules but executed without a shell.

    Example:
        RipgrepTool(search).run(request)  # raw_input = "TODO src"

    """

    def __init__(
        self, search: RipgrepSearch, timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    ) -> None:
        """Initialize the RipgrepTool with a search backend and timeout.

        Example:
            tool = RipgrepTool(search, timeout_seconds=15.0)

        """
        self._search = search
        self._timeout_seconds = timeout_seconds

    @property
    def spec(self) -> ToolSpec:
        """The tool's schema, name, and examples.

        Example:
            spec = tool.spec

        """
        return ToolSpec(
            ToolName("ripgrep"),
            "Search file contents with ripgrep and return matching lines.",
            ToolInputSchema(
                "A ripgrep argument line: a regex pattern, then optional paths "
                "and flags.",
                ToolExamples(("TODO src", "-i error logs", '"def main" -t py')),
                {"type": "string"},
            ),
        )

    def run(self, request: ToolRunRequest) -> ToolRunResult:
        """Run the ripgrep search tool against the local filesystem.

        Example:
            result = tool.run(ToolRunRequest(ToolName("ripgrep"), ToolInput("TODO")))

        """
        try:
            arguments = shlex.split(request.raw_input.value)
        except ValueError as error:
            return ToolRunResult(
                request.tool_name,
                ToolOutput(f"ripgrep error: {error}"),
                succeeded=False,
            )

        outcome = self._search.run(arguments, self._timeout_seconds)
        return self._to_result(request.tool_name, outcome)

    def _to_result(self, tool_name: ToolName, outcome: RipgrepOutcome) -> ToolRunResult:
        if outcome.exit_code is None:
            return ToolRunResult(tool_name, ToolOutput(outcome.stderr), succeeded=False)

        if outcome.exit_code == 0:
            return ToolRunResult(tool_name, ToolOutput(outcome.stdout), succeeded=True)

        if outcome.exit_code == NO_MATCH_EXIT_CODE:
            return ToolRunResult(
                tool_name, ToolOutput("No matches found."), succeeded=True
            )

        return ToolRunResult(tool_name, ToolOutput(outcome.stderr), succeeded=False)
