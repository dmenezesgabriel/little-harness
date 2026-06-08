"""Tests for entry-point plugin discovery, the one dynamic-import seam."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import pytest
from little_harness.application.ports.agent_tool import AgentTool
from little_harness.application.ports.chat_model import ChatModel
from little_harness.domain.errors import (
    UnknownObserverError,
    UnknownPolicyError,
    UnknownProviderError,
    UnknownToolError,
    UnknownUiError,
)
from little_harness.domain.tool_result import ToolRunRequest, ToolRunResult
from little_harness.domain.tool_spec import ToolInputSchema, ToolSpec
from little_harness.domain.values.text_values import ToolName, ToolOutput
from little_harness.plugin_discovery import (
    OBSERVER_GROUP,
    POLICY_GROUP,
    PROVIDER_GROUP,
    REPL_COMMAND_GROUP,
    TOOL_GROUP,
    UI_GROUP,
    ChatModelBuilder,
    default_policy_name,
    default_provider_name,
    discover_observer,
    discover_policy,
    discover_repl_commands,
    discover_tools,
    discover_ui,
    installed_providers,
    installed_tools,
    load_chat_model_builder,
)

from tests.plugin_fakes import (
    FakeAgentPolicy,
    FakeChatModel,
    FakeEntryPoint,
    FakeObserver,
    install_entry_points,
    make_observer_builder,
    make_policy_builder,
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


class TestDiscoverPolicy:
    def test_builds_the_policy_registered_under_the_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        install_entry_points(
            monkeypatch, {POLICY_GROUP: [FakeEntryPoint("json", make_policy_builder())]}
        )

        # Act / Assert
        assert isinstance(discover_policy("json"), FakeAgentPolicy)

    def test_rejects_an_unknown_policy_and_lists_installed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        install_entry_points(
            monkeypatch, {POLICY_GROUP: [FakeEntryPoint("json", make_policy_builder())]}
        )

        # Act / Assert: the message names the offending value and what is installed.
        with pytest.raises(UnknownPolicyError) as err:
            discover_policy("mystery")
        assert str(err.value) == (
            "Unknown policy: 'mystery'. Installed policy plugins: ['json']."
        )


class TestDefaultPolicyName:
    def test_returns_the_sole_installed_policy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        install_entry_points(
            monkeypatch, {POLICY_GROUP: [FakeEntryPoint("json", make_policy_builder())]}
        )

        # Act / Assert
        assert default_policy_name() == "json"

    def test_rejects_when_no_policy_is_installed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        install_entry_points(monkeypatch, {})

        # Act / Assert: the message names the count and the installed list.
        with pytest.raises(UnknownPolicyError) as err:
            default_policy_name()
        assert str(err.value) == (
            "No policy selected and 0 installed: []. "
            "Pass --policy, or install exactly one policy plugin."
        )

    def test_rejects_when_several_policies_are_installed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        install_entry_points(
            monkeypatch,
            {
                POLICY_GROUP: [
                    FakeEntryPoint("json", make_policy_builder()),
                    FakeEntryPoint("react", make_policy_builder()),
                ]
            },
        )

        # Act / Assert: ambiguous, so the sorted installed list is reported.
        with pytest.raises(UnknownPolicyError) as err:
            default_policy_name()
        assert str(err.value) == (
            "No policy selected and 2 installed: ['json', 'react']. "
            "Pass --policy, or install exactly one policy plugin."
        )


class TestDiscoverObserver:
    def test_builds_the_observer_registered_under_the_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        install_entry_points(
            monkeypatch,
            {OBSERVER_GROUP: [FakeEntryPoint("logging", make_observer_builder())]},
        )

        # Act / Assert
        assert isinstance(discover_observer("logging"), FakeObserver)

    def test_rejects_an_unknown_observer_and_lists_installed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        install_entry_points(
            monkeypatch,
            {OBSERVER_GROUP: [FakeEntryPoint("logging", make_observer_builder())]},
        )

        # Act / Assert
        with pytest.raises(UnknownObserverError) as err:
            discover_observer("mystery")
        assert str(err.value) == (
            "Unknown observer: 'mystery'. Installed observer plugins: ['logging']."
        )


class TestDiscoverReplCommands:
    def test_discover_repl_commands_loads_all_registered(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class FakeCommand:
            name = "custom"
            aliases = ()
            description = "Custom test command"

            def execute(self, console: object, /) -> None:
                pass

        install_entry_points(
            monkeypatch,
            {REPL_COMMAND_GROUP: [FakeEntryPoint("custom_cmd", FakeCommand)]},
        )

        commands = discover_repl_commands()
        assert len(commands) == 1
        assert commands[0].name == "custom"


class TestDiscoverUi:
    def test_builds_the_ui_registered_under_the_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        def fake_builder(_app: object, _reg: object) -> str:
            return "fake_ui"

        install_entry_points(
            monkeypatch, {UI_GROUP: [FakeEntryPoint("rich", fake_builder)]}
        )

        # Act / Assert
        assert discover_ui("rich") is fake_builder

    def test_rejects_an_unknown_ui_and_lists_installed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        def fake_builder(_app: object, _reg: object) -> str:
            return "fake_ui"

        install_entry_points(
            monkeypatch, {UI_GROUP: [FakeEntryPoint("rich", fake_builder)]}
        )

        # Act / Assert: the message names the offending value and what is installed.
        with pytest.raises(UnknownUiError) as err:
            discover_ui("mystery")
        assert str(err.value) == (
            "Unknown UI: 'mystery'. Installed UI plugins: ['rich']."
        )
