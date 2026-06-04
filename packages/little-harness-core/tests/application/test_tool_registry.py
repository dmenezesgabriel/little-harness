from __future__ import annotations

import pytest
from little_harness.application.tool_registry import ToolRegistry
from little_harness.domain.values.text_values import ToolName

from tests.application.fakes import RecordingAgentTool


class TestToolRegistry:
    def test_finds_a_registered_tool_by_name(self) -> None:
        # Arrange
        tool = RecordingAgentTool()
        registry = ToolRegistry([tool])

        # Act / Assert
        assert registry.find(ToolName("calculator")) is tool
        assert registry.find(ToolName("missing")) is None
        assert len(registry) == 1

    def test_exposes_specs_of_registered_tools(self) -> None:
        # Arrange
        registry = ToolRegistry([RecordingAgentTool()])

        # Act
        specs = registry.specs()

        # Assert
        assert [spec.name for spec in specs] == [ToolName("calculator")]

    def test_rejects_duplicate_tool_names(self) -> None:
        # Act / Assert
        with pytest.raises(ValueError, match="Duplicate tool name"):
            ToolRegistry([RecordingAgentTool(), RecordingAgentTool()])
