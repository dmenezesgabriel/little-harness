from __future__ import annotations

import pytest

from local_llm.domain.values.text_values import (
    MessageContent,
    Prompt,
    RunId,
    ToolInput,
    ToolName,
    ToolOutput,
)


class TestPrompt:
    def test_keeps_non_empty_value(self) -> None:
        # Act
        prompt = Prompt("What is 2 + 2?")

        # Assert
        assert prompt.value == "What is 2 + 2?"

    @pytest.mark.parametrize("blank", ["", "   ", "\n\t"])
    def test_rejects_blank_value(self, blank: str) -> None:
        # Act / Assert
        with pytest.raises(ValueError, match="Prompt is empty"):
            Prompt(blank)


class TestToolName:
    def test_trims_surrounding_whitespace(self) -> None:
        # Act
        name = ToolName("  calculator  ")

        # Assert
        assert name.value == "calculator"

    def test_rejects_blank_value(self) -> None:
        # Act / Assert
        with pytest.raises(ValueError, match="ToolName is empty"):
            ToolName("   ")


class TestToolInput:
    def test_trims_but_allows_empty(self) -> None:
        # Act
        trimmed = ToolInput("  144 / 12  ")
        empty = ToolInput("   ")

        # Assert
        assert trimmed.value == "144 / 12"
        assert empty.value == ""


class TestRunId:
    def test_keeps_non_empty_value(self) -> None:
        # Act / Assert
        assert RunId("a1b2c3").value == "a1b2c3"

    def test_rejects_blank_value(self) -> None:
        # Act / Assert
        with pytest.raises(ValueError, match="RunId is empty"):
            RunId("   ")


class TestPlainTextWrappers:
    def test_message_content_wraps_value_verbatim(self) -> None:
        # Act / Assert
        assert MessageContent("  spaced  ").value == "  spaced  "

    def test_tool_output_wraps_value_verbatim(self) -> None:
        # Act / Assert
        assert ToolOutput("12").value == "12"
