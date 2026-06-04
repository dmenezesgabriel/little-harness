from __future__ import annotations

from little_harness.domain.tool_result import ToolRunRequest
from little_harness.domain.values.text_values import ToolInput, ToolName
from little_harness_ripgrep.ripgrep_search import RipgrepOutcome
from little_harness_ripgrep.ripgrep_tool import RipgrepTool

from tests.unit.fakes import FakeRipgrepSearch


def ripgrep_request(arguments: str) -> ToolRunRequest:
    return ToolRunRequest(ToolName("ripgrep"), ToolInput(arguments))


def tool_returning(outcome: RipgrepOutcome) -> tuple[RipgrepTool, FakeRipgrepSearch]:
    search = FakeRipgrepSearch(outcome)
    return RipgrepTool(search, timeout_seconds=15.0), search


class TestRipgrepTool:
    def test_advertises_a_safe_spec(self) -> None:
        # Act
        spec = RipgrepTool(FakeRipgrepSearch(RipgrepOutcome(0, "", ""))).spec

        # Assert: searching is read-only, so it never asks for approval.
        assert spec.name == ToolName("ripgrep")
        assert spec.requires_approval is False

    def test_returns_matches_and_tokenizes_arguments(self) -> None:
        # Arrange
        tool, search = tool_returning(RipgrepOutcome(0, "app.py:1:TODO\n", ""))

        # Act: a quoted pattern with a path is split with shell rules.
        result = tool.run(ripgrep_request('"to do" src'))

        # Assert
        assert result.tool_name == ToolName("ripgrep")
        assert result.succeeded is True
        assert result.output.value == "app.py:1:TODO\n"
        assert search.argument_calls == [["to do", "src"]]
        assert search.timeouts == [15.0]

    def test_reports_no_matches_as_a_success(self) -> None:
        # Arrange: ripgrep exit code 1 means "no matches", not an error.
        tool, _ = tool_returning(RipgrepOutcome(1, "", ""))

        # Act
        result = tool.run(ripgrep_request("missing"))

        # Assert
        assert result.tool_name == ToolName("ripgrep")
        assert result.succeeded is True
        assert result.output.value == "No matches found."

    def test_reports_a_search_error_as_a_failure(self) -> None:
        # Arrange: exit code 2+ is a real ripgrep error.
        tool, _ = tool_returning(RipgrepOutcome(2, "", "regex parse error\n"))

        # Act
        result = tool.run(ripgrep_request("("))

        # Assert
        assert result.tool_name == ToolName("ripgrep")
        assert result.succeeded is False
        assert result.output.value == "regex parse error\n"

    def test_reports_a_missing_binary_as_a_failure(self) -> None:
        # Arrange: a None exit code means rg was absent or timed out.
        tool, _ = tool_returning(RipgrepOutcome(None, "", "ripgrep not found"))

        # Act
        result = tool.run(ripgrep_request("anything"))

        # Assert
        assert result.tool_name == ToolName("ripgrep")
        assert result.succeeded is False
        assert result.output.value == "ripgrep not found"

    def test_reports_unbalanced_quotes_as_a_failure(self) -> None:
        # Arrange
        tool, search = tool_returning(RipgrepOutcome(0, "", ""))

        # Act: an unterminated quote cannot be tokenized.
        result = tool.run(ripgrep_request('"unterminated'))

        # Assert: it never reaches the search boundary.
        assert result.tool_name == ToolName("ripgrep")
        assert result.succeeded is False
        assert result.output.value.startswith("ripgrep error:")
        assert search.argument_calls == []
