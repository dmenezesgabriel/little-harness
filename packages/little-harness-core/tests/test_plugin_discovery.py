"""Tests for entry-point plugin discovery, the one dynamic-import seam."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import pytest
from little_harness.application.ports.agent_tool import AgentTool
from little_harness.application.ports.chat_model import ChatModel
from little_harness.domain.errors import UnknownProviderError, UnknownToolError
from little_harness.domain.tool_result import ToolRunRequest, ToolRunResult
from little_harness.domain.tool_spec import ToolInputSchema, ToolSpec
from little_harness.domain.values.text_values import ToolName, ToolOutput
from little_harness.plugin_discovery import (
    PROVIDER_GROUP,
    TOOL_GROUP,
    ChatModelBuilder,
    default_provider_name,
    discover_tools,
    installed_providers,
    installed_tools,
    load_chat_model_builder,
)

from tests.plugin_fakes import (
    FakeChatModel,
    FakeEntryPoint,
    install_entry_points,
    make_provider_builder,
)


class FakeTool:
    """Minimal AgentTool used to assert tool discovery wiring."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(ToolName("fake"), "A fake tool.", ToolInputSchema("input"))

    def run(self, request: ToolRunRequest) -> ToolRunResult:
        return ToolRunResult(request.tool_name, ToolOutput("ok"), succeeded=True)


def unbuilt_provider(_options: Mapping[str, str]) -> ChatModel:
    return FakeChatModel("")


class NamedFakeTool:
    """AgentTool whose advertised name is set per instance, for selection tests."""

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(ToolName(self._name), "A fake tool.", ToolInputSchema("input"))

    def run(self, request: ToolRunRequest) -> ToolRunResult:
        return ToolRunResult(request.tool_name, ToolOutput("ok"), succeeded=True)


def make_tool_builder(name: str) -> Callable[[], AgentTool]:
    def build() -> AgentTool:
        return NamedFakeTool(name)

    return build


def install_three_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    install_entry_points(
        monkeypatch,
        {
            TOOL_GROUP: [
                FakeEntryPoint("read_file", make_tool_builder("read_file")),
                FakeEntryPoint("bash", make_tool_builder("bash")),
                FakeEntryPoint("ripgrep", make_tool_builder("ripgrep")),
            ]
        },
    )


class TestLoadChatModelBuilder:
    def test_returns_the_builder_registered_under_the_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        builder: ChatModelBuilder = unbuilt_provider
        install_entry_points(
            monkeypatch, {PROVIDER_GROUP: [FakeEntryPoint("llama_cpp", builder)]}
        )

        # Act / Assert
        assert load_chat_model_builder("llama_cpp") is builder

    def test_rejects_an_unknown_provider_and_lists_installed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        install_entry_points(
            monkeypatch,
            {PROVIDER_GROUP: [FakeEntryPoint("llama_cpp", unbuilt_provider)]},
        )

        # Act / Assert: the message names the offending value and what is installed.
        with pytest.raises(UnknownProviderError, match="Unknown provider: 'litellm'"):
            load_chat_model_builder("litellm")

    def test_rejects_a_provider_whose_target_is_not_callable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        install_entry_points(
            monkeypatch, {PROVIDER_GROUP: [FakeEntryPoint("broken", "not-callable")]}
        )

        # Act / Assert: the exact message names the offending provider and value.
        with pytest.raises(UnknownProviderError) as err:
            load_chat_model_builder("broken")
        assert str(err.value) == (
            "Provider 'broken' did not load a callable builder: 'not-callable'."
        )


class TestInstalledProviders:
    def test_returns_sorted_provider_names(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        install_entry_points(
            monkeypatch,
            {
                PROVIDER_GROUP: [
                    FakeEntryPoint("litellm", make_provider_builder("")),
                    FakeEntryPoint("llama_cpp", make_provider_builder("")),
                ]
            },
        )

        # Act / Assert
        assert installed_providers() == ["litellm", "llama_cpp"]


class TestDefaultProviderName:
    def test_returns_the_sole_installed_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        install_entry_points(
            monkeypatch,
            {PROVIDER_GROUP: [FakeEntryPoint("llama_cpp", unbuilt_provider)]},
        )

        # Act / Assert
        assert default_provider_name() == "llama_cpp"

    def test_rejects_when_no_provider_is_installed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        install_entry_points(monkeypatch, {})

        # Act / Assert: the message names the count and the installed list.
        with pytest.raises(UnknownProviderError) as err:
            default_provider_name()
        assert str(err.value) == (
            "No provider selected and 0 installed: []. "
            "Pass --provider, or install exactly one provider plugin."
        )

    def test_rejects_when_several_providers_are_installed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        install_entry_points(
            monkeypatch,
            {
                PROVIDER_GROUP: [
                    FakeEntryPoint("litellm", unbuilt_provider),
                    FakeEntryPoint("llama_cpp", unbuilt_provider),
                ]
            },
        )

        # Act / Assert: ambiguous, so the sorted installed list is reported.
        with pytest.raises(UnknownProviderError) as err:
            default_provider_name()
        assert str(err.value) == (
            "No provider selected and 2 installed: ['litellm', 'llama_cpp']. "
            "Pass --provider, or install exactly one provider plugin."
        )


class TestDiscoverTools:
    def test_instantiates_each_registered_tool(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        install_entry_points(
            monkeypatch, {TOOL_GROUP: [FakeEntryPoint("fake", FakeTool)]}
        )

        # Act
        tools = discover_tools()

        # Assert
        assert [tool.spec.name for tool in tools] == [ToolName("fake")]

    def test_returns_empty_when_no_tools_are_registered(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        install_entry_points(monkeypatch, {})

        # Act / Assert
        assert discover_tools() == []

    def test_returns_every_tool_sorted_by_name_when_no_selection(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        install_three_tools(monkeypatch)

        # Act
        names = [tool.spec.name.value for tool in discover_tools()]

        # Assert: deterministic order regardless of entry-point registration order.
        assert names == ["bash", "read_file", "ripgrep"]

    def test_limits_discovery_to_the_selected_names_in_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        install_three_tools(monkeypatch)

        # Act
        names = [tool.spec.name.value for tool in discover_tools(["ripgrep", "bash"])]

        # Assert: selection order is preserved, unselected tools are dropped.
        assert names == ["ripgrep", "bash"]

    def test_rejects_a_selected_tool_that_is_not_installed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        install_three_tools(monkeypatch)

        # Act / Assert: the message names the offending value and the installed list.
        with pytest.raises(UnknownToolError) as err:
            discover_tools(["read_file", "mystery"])
        assert str(err.value) == (
            "Unknown tool: 'mystery'. "
            "Installed tools: ['bash', 'read_file', 'ripgrep']."
        )


class TestInstalledTools:
    def test_returns_sorted_tool_names(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Arrange
        install_three_tools(monkeypatch)

        # Act / Assert
        assert installed_tools() == ["bash", "read_file", "ripgrep"]
