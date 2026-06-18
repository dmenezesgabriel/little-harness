from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

import pytest
from little_harness.application.agent_runtime import AgentRuntimeConfig
from little_harness.application.hook_chain import HookChain
from little_harness.application.ports.agent_observer import AgentObserver
from little_harness.application.ports.agent_tool import AgentTool
from little_harness.application.ports.chat_model import ChatModel
from little_harness.application.ports.lifecycle_hook import LifecycleHook
from little_harness.application.ports.permission_requester import PermissionRequester
from little_harness.application.ports.session_plugin import SessionPlugin
from little_harness.application.tool_registry import ToolRegistry
from little_harness.composition import (
    _build_session_plugin,
    build_application,
    build_chat_model,
    build_command_registry,
    build_dependencies,
    build_hook_list,
    build_hooks,
    build_observer,
    build_permission_requester,
    build_policy,
    build_token_sink,
    run_cli,
    to_runtime_config,
)
from little_harness.domain.decision import FinalAnswer, ToolCall
from little_harness.domain.errors import (
    UnknownPermissionRequesterError,
    UnknownPolicyError,
    UnknownProviderError,
)
from little_harness.domain.hook_decision import Proceed
from little_harness.domain.message import ChatMessage
from little_harness.domain.message_history import MessageHistory
from little_harness.domain.result import AgentResult
from little_harness.domain.steps import AgentSteps
from little_harness.domain.tool_result import ToolRunRequest, ToolRunResult
from little_harness.domain.tool_spec import ToolInputSchema, ToolSpec
from little_harness.domain.values.numeric_values import (
    ElapsedSeconds,
    Iteration,
    MaxIterations,
    MaxTokens,
    RepeatPenalty,
    Temperature,
    TopP,
)
from little_harness.domain.values.role import SYSTEM, USER
from little_harness.domain.values.text_values import (
    MessageContent,
    Prompt,
    RunId,
    SessionId,
    ToolInput,
    ToolName,
    ToolOutput,
)
from little_harness.infrastructure.hooks.approval_hook import ApprovalHook
from little_harness.infrastructure.observability.null_observer import NullObserver
from little_harness.infrastructure.permissions.auto_approve_requester import (
    AutoApprovePermissionRequester,
)
from little_harness.plugin_discovery import (
    OBSERVER_GROUP,
    POLICY_GROUP,
    PROVIDER_GROUP,
    REPL_COMMAND_GROUP,
    SESSION_PLUGIN_GROUP,
)
from little_harness.presentation.cli.app_config import AppConfig
from little_harness.presentation.cli.argument_parser import ArgumentParser
from little_harness.presentation.cli.permission_prompt import (
    InteractivePermissionRequester,
)
from little_harness.presentation.cli.token_sinks import NullTokenSink, StdoutTokenSink

from tests.application.fakes import RecordingObserver
from tests.plugin_fakes import (
    FakeChatModel,
    FakeEntryPoint,
    FakeObserver,
    FakeSessionPlugin,
    install_entry_points,
    make_observer_builder,
    make_policy_builder,
    make_provider_builder,
)

FINAL_ANSWER_JSON = (
    '{"action":"final","tool_name":null,"tool_input":null,'
    '"answer":"hello from the agent"}'
)


@pytest.fixture
def created_models(monkeypatch: pytest.MonkeyPatch) -> list[FakeChatModel]:
    """Register a fake provider and policy, and capture each model built.

    Core ships neither a provider nor a policy, so a full run needs both
    discoverable; the loop ends on the first reply via the fake policy.
    """
    created: list[FakeChatModel] = []

    def build(_options: Mapping[str, str]) -> ChatModel:
        model = FakeChatModel(FINAL_ANSWER_JSON)
        created.append(model)
        return model

    install_entry_points(
        monkeypatch,
        {
            PROVIDER_GROUP: [FakeEntryPoint("llama_cpp", build)],
            POLICY_GROUP: [FakeEntryPoint("json", make_policy_builder())],
        },
    )
    return created


@pytest.mark.usefixtures("created_models")
class TestComposition:
    def test_run_cli_wires_the_whole_stack_and_renders_the_answer(self) -> None:
        # Act
        output = run_cli(["--prompt", "hi"])

        # Assert
        assert "hello from the agent" in output
        assert "Elapsed:" in output

    def test_build_application_threads_the_observer_through_the_stack(self) -> None:
        # Arrange
        observer = RecordingObserver()
        config = ArgumentParser().parse(["--prompt", "hi"])

        # Act
        build_application(config, observer).run(Prompt("hi"))

        # Assert: the seam reaches a real run end-to-end.
        assert observer.events[0] == "run_started:hi"
        assert observer.events[-1] == "run_finished"
        assert len(observer.finished) == 1

    def test_build_system_message_returns_message_with_system_role(
        self, created_models: list[FakeChatModel]
    ) -> None:
        config = ArgumentParser().parse(["--prompt", "hi"])
        app = build_application(config)

        message = app.build_system_message()

        assert message.role == SYSTEM

    def test_run_turn_delegates_to_runtime_and_returns_updated_history(
        self, created_models: list[FakeChatModel]
    ) -> None:
        config = ArgumentParser().parse(["--prompt", "hi"])
        app = build_application(config)
        system = app.build_system_message()
        history = MessageHistory().with_message(system)

        result, updated = app.run_turn(Prompt("hello"), history)

        assert "hello from the agent" in result.answer.value
        assert len(list(updated)) == 3


class TestModelLifecycle:
    def test_run_cli_closes_the_model_after_the_run(
        self, created_models: list[FakeChatModel]
    ) -> None:
        # Act
        run_cli(["--prompt", "hi"])

        # Assert
        assert created_models and created_models[0].closed is True


class FakeApplication:
    """Stand-in for the Application context manager, recording its run output."""

    def __enter__(self) -> FakeApplication:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        return None

    def run(self, _prompt: Prompt) -> str:
        return "rendered"


class TestCompositionThreadsConfigThroughTheStack:
    """Each builder must pass its config-derived argument, not a placeholder.

    These pin the wiring itself: that the configured observer name, tool
    selection, config object, requester, and approval names actually reach the
    collaborator, so dropping any of them is caught.
    """

    def test_build_observer_passes_the_configured_name_to_discovery(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange: spy on discovery so the threaded name is observable.
        received: list[str] = []

        def fake_discover_observer(name: str) -> AgentObserver:
            received.append(name)
            return FakeObserver()

        monkeypatch.setattr(
            "little_harness.composition.discover_observer", fake_discover_observer
        )
        config = ArgumentParser().parse(["--prompt", "hi", "--observer", "logging"])

        # Act
        build_observer(config)

        # Assert
        assert received == ["logging"]

    @pytest.mark.usefixtures("created_models")
    def test_build_dependencies_selects_tools_from_the_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        received: list[tuple[str, ...] | None] = []

        def fake_discover_tools(selection: tuple[str, ...] | None) -> list[AgentTool]:
            received.append(selection)
            return []

        monkeypatch.setattr(
            "little_harness.composition.discover_tools", fake_discover_tools
        )
        config = ArgumentParser().parse(["--prompt", "hi", "--tools", "calculator"])

        # Act
        build_dependencies(config, NullObserver())

        # Assert
        assert received == [("calculator",)]

    @pytest.mark.usefixtures("created_models")
    def test_build_dependencies_passes_the_config_to_build_hooks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        received: list[AppConfig] = []

        def fake_build_hooks(
            _registry: ToolRegistry, config: AppConfig, _extra: object = None
        ) -> HookChain:
            received.append(config)
            return HookChain([])

        monkeypatch.setattr("little_harness.composition.build_hooks", fake_build_hooks)
        config = ArgumentParser().parse(["--prompt", "hi"])

        # Act
        build_dependencies(config, NullObserver())

        # Assert
        assert received == [config]

    def test_build_hooks_passes_the_config_to_build_hook_list(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        received: list[AppConfig] = []

        def fake_build_hook_list(
            _registry: ToolRegistry, config: AppConfig, _extra: object = None
        ) -> list[LifecycleHook]:
            received.append(config)
            return []

        monkeypatch.setattr(
            "little_harness.composition.build_hook_list", fake_build_hook_list
        )
        config = ArgumentParser().parse(["--prompt", "hi", "--yes"])

        # Act
        build_hooks(ToolRegistry([]), config)

        # Assert
        assert received == [config]

    def test_build_hook_list_wires_the_approval_hook_with_requester_and_names(
        self,
    ) -> None:
        # Arrange: a sensitive tool and --yes (auto-approve).
        config = ArgumentParser().parse(["--prompt", "hi", "--yes"])
        hooks = build_hook_list(ToolRegistry([SensitiveTool()]), config)

        # Act: gating a sensitive call exercises both the requester and the names.
        decision = hooks[0].on_pre_tool_use(
            RunId("r"), Iteration(1), ToolCall(ToolName("bash"), ToolInput("ls"))
        )

        # Assert: a missing requester or missing names would raise here instead.
        assert decision == Proceed()

    def test_run_cli_threads_the_built_observer_into_the_application(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange: a recognizable observer, and a fake application that records it.
        sentinel = RecordingObserver()
        received: list[AgentObserver] = []

        def fake_build_observer(_config: AppConfig) -> AgentObserver:
            return sentinel

        def fake_build_application(
            _config: AppConfig,
            observer: AgentObserver,
            _skill_loader: object = None,
            _extra_hooks: object = None,
        ) -> FakeApplication:
            received.append(observer)
            return FakeApplication()

        monkeypatch.setattr(
            "little_harness.composition.build_observer", fake_build_observer
        )
        monkeypatch.setattr(
            "little_harness.composition.build_application", fake_build_application
        )

        # Act
        output = run_cli(["--prompt", "hi"])

        # Assert: the built observer reaches the application, and its run renders.
        assert output == "rendered"
        assert received == [sentinel]


class TestRunCliInteractive:
    def test_returns_empty_string_and_starts_interactive_console_when_no_prompt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        started: list[bool] = []

        def fake_start(_self: object) -> str:
            started.append(True)
            return ""

        def fake_build_app(
            _config: object,
            _observer: object,
            _skill_loader: object = None,
            _extra_hooks: object = None,
        ) -> FakeApplication:
            return FakeApplication()

        def fake_build_obs(_config: object) -> None:
            return None

        monkeypatch.setattr(
            "little_harness.presentation.cli.interactive_console.InteractiveConsole.start",
            fake_start,
        )
        monkeypatch.setattr(
            "little_harness.composition.build_application",
            fake_build_app,
        )
        monkeypatch.setattr("little_harness.composition.build_observer", fake_build_obs)
        monkeypatch.setattr(
            "little_harness.composition._build_session_plugin",
            lambda _: None,  # type: ignore[reportUnknownLambdaType]
        )

        result = run_cli([])

        assert result == ""
        assert started == [True]

    def test_invokes_build_application_when_no_prompt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        built: list[object] = []

        def fake_build(
            _config: AppConfig,
            _observer: AgentObserver,
            _skill_loader: object = None,
            _extra_hooks: object = None,
        ) -> FakeApplication:
            built.append(_config)
            return FakeApplication()

        def fake_start_console(_self: object) -> str:
            return ""

        def fake_build_obs(_config: object) -> None:
            return None

        monkeypatch.setattr(
            "little_harness.presentation.cli.interactive_console.InteractiveConsole.start",
            fake_start_console,
        )
        monkeypatch.setattr("little_harness.composition.build_application", fake_build)
        monkeypatch.setattr("little_harness.composition.build_observer", fake_build_obs)
        monkeypatch.setattr(
            "little_harness.composition._build_session_plugin",
            lambda _: None,  # type: ignore[reportUnknownLambdaType]
        )

        run_cli([])

        assert len(built) == 1

    def test_invokes_build_observer_when_no_prompt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        built: list[object] = []

        def fake_build(_config: AppConfig) -> AgentObserver:
            built.append(_config)
            return RecordingObserver()

        def fake_start_console(_self: object) -> str:
            return ""

        def fake_build_app(
            _config: object,
            _observer: object,
            _skill_loader: object = None,
            _extra_hooks: object = None,
        ) -> FakeApplication:
            return FakeApplication()

        monkeypatch.setattr(
            "little_harness.presentation.cli.interactive_console.InteractiveConsole.start",
            fake_start_console,
        )
        monkeypatch.setattr(
            "little_harness.composition.build_application",
            fake_build_app,
        )
        monkeypatch.setattr("little_harness.composition.build_observer", fake_build)
        monkeypatch.setattr(
            "little_harness.composition._build_session_plugin",
            lambda _: None,  # type: ignore[reportUnknownLambdaType]
        )

        run_cli([])

        assert len(built) == 1

    def test_passes_built_application_to_interactive_console(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class SpyingInteractiveConsole:
            received_app: object = None
            received_registry: object = None

            def __init__(
                self,
                app: object,
                output: object = None,
                source: object = None,
                registry: object = None,
                skill_loader: object = None,
                _initial_messages: object = None,
            ) -> None:
                SpyingInteractiveConsole.received_app = app
                SpyingInteractiveConsole.received_registry = registry

            def start(self) -> str:
                return ""

        def fake_build_app(
            _config: object,
            _observer: object,
            _skill_loader: object = None,
            _extra_hooks: object = None,
        ) -> FakeApplication:
            return built_app

        def fake_build_obs(_config: object) -> None:
            return None

        built_app = FakeApplication()
        monkeypatch.setattr(
            "little_harness.composition.build_application",
            fake_build_app,
        )
        monkeypatch.setattr("little_harness.composition.build_observer", fake_build_obs)
        monkeypatch.setattr(
            "little_harness.composition.InteractiveConsole",
            SpyingInteractiveConsole,
        )
        monkeypatch.setattr(
            "little_harness.composition._build_session_plugin",
            lambda _: None,  # type: ignore[reportUnknownLambdaType]
        )

        run_cli([])

        assert SpyingInteractiveConsole.received_app is built_app
        assert SpyingInteractiveConsole.received_registry is not None

    def test_starts_custom_ui_when_selected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        passed_args: list[tuple[object, object]] = []

        class FakeCustomUi:
            def __init__(self, app: object, registry: object) -> None:
                passed_args.append((app, registry))

            def start(self) -> str:
                return "custom_started"

        def fake_build_app(
            _config: object,
            _observer: object,
            _skill_loader: object = None,
            _extra_hooks: object = None,
        ) -> FakeApplication:
            return built_app

        def fake_build_obs(_config: object) -> None:
            return None

        def fake_discover_ui(name: str) -> type[FakeCustomUi] | None:
            return FakeCustomUi if name == "custom" else None

        built_app = FakeApplication()
        monkeypatch.setattr(
            "little_harness.composition.build_application",
            fake_build_app,
        )
        monkeypatch.setattr("little_harness.composition.build_observer", fake_build_obs)
        monkeypatch.setattr(
            "little_harness.composition.discover_ui",
            fake_discover_ui,
        )
        monkeypatch.setattr(
            "little_harness.composition._build_session_plugin",
            lambda _: None,  # type: ignore[reportUnknownLambdaType]
        )

        result = run_cli(["--ui", "custom"])

        assert result == "custom_started"
        assert len(passed_args) == 1
        assert passed_args[0][0] is built_app
        assert passed_args[0][1] is not None

    def test_build_command_registry_includes_discovered_repl_commands_with_source(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class FakeReplCommand:
            name = "customcmd"
            aliases = ()

        monkeypatch.setattr(
            "little_harness.composition.discover_repl_commands",
            lambda: [FakeReplCommand()],
        )

        registry = build_command_registry()
        assert "/customcmd" in registry._index
        assert registry._sources["/customcmd"] == f"plugin:{FakeReplCommand.__module__}"


class TestProviderSelection:
    def test_builds_the_selected_provider_via_discovery(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        install_entry_points(
            monkeypatch,
            {PROVIDER_GROUP: [FakeEntryPoint("llama_cpp", make_provider_builder(""))]},
        )
        config = ArgumentParser().parse(["--prompt", "hi"])

        # Act / Assert
        assert isinstance(build_chat_model(config), FakeChatModel)

    def test_rejects_no_selection_when_several_providers_installed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange: two providers, and the config names none (provider stays None).
        install_entry_points(
            monkeypatch,
            {
                PROVIDER_GROUP: [
                    FakeEntryPoint("litellm", make_provider_builder("")),
                    FakeEntryPoint("llama_cpp", make_provider_builder("")),
                ]
            },
        )
        config = ArgumentParser().parse(["--prompt", "hi"])

        # Act / Assert: ambiguous default fails instead of guessing.
        with pytest.raises(UnknownProviderError, match=r"2 installed"):
            build_chat_model(config)

    def test_rejects_an_unknown_provider_with_the_installed_list(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        install_entry_points(
            monkeypatch,
            {PROVIDER_GROUP: [FakeEntryPoint("llama_cpp", make_provider_builder(""))]},
        )
        config = ArgumentParser().parse(["--prompt", "hi", "--provider", "mystery"])

        # Act / Assert
        with pytest.raises(
            UnknownProviderError, match=r"Unknown provider: 'mystery'.*\['llama_cpp'\]"
        ):
            build_chat_model(config)

    def test_passes_provider_options_to_the_builder(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange: capture the options the selected builder receives.
        received: list[Mapping[str, str]] = []

        def build(options: Mapping[str, str]) -> ChatModel:
            received.append(options)
            return FakeChatModel("")

        install_entry_points(
            monkeypatch, {PROVIDER_GROUP: [FakeEntryPoint("llama_cpp", build)]}
        )
        config = ArgumentParser().parse(["--prompt", "hi", "-o", "model_path=/m.gguf"])

        # Act
        build_chat_model(config)

        # Assert
        assert received == [{"model_path": "/m.gguf"}]


class TestPolicySelection:
    def test_builds_the_selected_policy_via_discovery(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        install_entry_points(
            monkeypatch, {POLICY_GROUP: [FakeEntryPoint("json", make_policy_builder())]}
        )
        config = ArgumentParser().parse(["--prompt", "hi"])

        # Act: the sole installed policy is selected when --policy is omitted.
        policy = build_policy(config)

        # Assert: the discovered fake policy is built (parses output as final).
        decision = policy.parse_model_output(MessageContent("anything"))
        assert decision == FinalAnswer(MessageContent("anything"))

    def test_uses_the_explicitly_selected_policy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange: a real policy is installed, but --policy names a missing one.
        install_entry_points(
            monkeypatch, {POLICY_GROUP: [FakeEntryPoint("json", make_policy_builder())]}
        )
        config = ArgumentParser().parse(["--prompt", "hi", "--policy", "mystery"])

        # Act / Assert: the explicit --policy wins over the sole-installed default.
        with pytest.raises(UnknownPolicyError, match="mystery"):
            build_policy(config)


class TestRuntimeConfig:
    def test_copies_the_sampling_and_loop_bounds(self) -> None:
        # Arrange
        config = ArgumentParser().parse(
            [
                "--prompt",
                "hi",
                "--temperature",
                "0.5",
                "--top-p",
                "0.9",
                "--repeat-penalty",
                "1.2",
                "--max-tokens",
                "99",
                "--max-iterations",
                "7",
            ]
        )

        # Act / Assert
        assert to_runtime_config(config) == AgentRuntimeConfig(
            max_iterations=MaxIterations(7),
            temperature=Temperature(0.5),
            max_tokens=MaxTokens(99),
            top_p=TopP(0.9),
            repeat_penalty=RepeatPenalty(1.2),
        )


class TestObserverSelection:
    def test_defaults_to_the_null_observer(self) -> None:
        # Arrange
        config = ArgumentParser().parse(["--prompt", "hi"])

        # Act / Assert
        assert isinstance(build_observer(config), NullObserver)

    def test_selects_the_named_observer_via_discovery(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        install_entry_points(
            monkeypatch,
            {OBSERVER_GROUP: [FakeEntryPoint("logging", make_observer_builder())]},
        )
        config = ArgumentParser().parse(["--prompt", "hi", "--observer", "logging"])

        # Act / Assert
        assert isinstance(build_observer(config), FakeObserver)

    def test_log_flag_is_shorthand_for_the_logging_observer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        install_entry_points(
            monkeypatch,
            {OBSERVER_GROUP: [FakeEntryPoint("logging", make_observer_builder())]},
        )
        config = ArgumentParser().parse(["--prompt", "hi", "--log"])

        # Act / Assert: --log resolves to the discovered `logging` observer.
        assert isinstance(build_observer(config), FakeObserver)


class TestTokenSinkSelection:
    def test_defaults_to_the_null_token_sink(self) -> None:
        # Arrange
        config = ArgumentParser().parse(["--prompt", "hi"])

        # Act / Assert
        assert isinstance(build_token_sink(config), NullTokenSink)

    def test_selects_stdout_sink_when_stream_flag_set(self) -> None:
        # Arrange
        config = ArgumentParser().parse(["--prompt", "hi", "--stream"])

        # Act / Assert
        assert isinstance(build_token_sink(config), StdoutTokenSink)


class SensitiveTool:
    """A fake tool that declares it needs human approval before running."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            ToolName("bash"),
            "Run a shell command.",
            ToolInputSchema("a command"),
            requires_approval=True,
        )

    def run(self, request: ToolRunRequest) -> ToolRunResult:
        return ToolRunResult(request.tool_name, ToolOutput("ran"), succeeded=True)


class TestHookSelection:
    def test_every_run_is_wrapped_in_a_hook_chain(self) -> None:
        # Act / Assert: the chain is the single composition point for hooks.
        config = ArgumentParser().parse(["--prompt", "hi", "--yes"])

        assert isinstance(build_hooks(ToolRegistry([]), config), HookChain)

    def test_adds_no_hook_when_no_tool_needs_approval(self) -> None:
        # Act / Assert: an empty chain folds to Proceed, like the null hook.
        config = ArgumentParser().parse(["--prompt", "hi", "--yes"])

        assert build_hook_list(ToolRegistry([]), config) == []

    def test_adds_an_approval_hook_when_a_tool_needs_approval(self) -> None:
        # Act
        config = ArgumentParser().parse(["--prompt", "hi", "--yes"])
        hooks = build_hook_list(ToolRegistry([SensitiveTool()]), config)

        # Assert
        assert len(hooks) == 1
        assert isinstance(hooks[0], ApprovalHook)


class TestPermissionRequesterSelection:
    def test_auto_approves_when_yes_flag_is_set(self) -> None:
        # Act / Assert: --yes runs unattended regardless of a terminal.
        config = ArgumentParser().parse(["--prompt", "hi", "--yes"])

        assert isinstance(
            build_permission_requester(config), AutoApprovePermissionRequester
        )

    def test_auto_approves_when_no_terminal_is_attached(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange: piped input has no tty, so there is no human to prompt.
        config = ArgumentParser().parse(["--prompt", "hi"])
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)

        # Act / Assert
        assert isinstance(
            build_permission_requester(config), AutoApprovePermissionRequester
        )

    def test_prompts_interactively_when_a_terminal_is_attached(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange: a real terminal and no --yes means a human is asked.
        config = ArgumentParser().parse(["--prompt", "hi"])
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)

        def assert_not_called(name: str) -> None:
            raise AssertionError(f"Should not try to discover {name}")

        monkeypatch.setattr(
            "little_harness.composition.discover_permission_requester",
            assert_not_called,
        )

        assert isinstance(
            build_permission_requester(config), InteractivePermissionRequester
        )

    def test_resolves_ui_specific_requester_when_available(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class FakeUiRequester(PermissionRequester):
            def request_approval(self, call: ToolCall, /) -> bool:
                return True

        config = ArgumentParser().parse(["--prompt", "hi", "--ui", "custom"])
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)

        def fake_discover(name: str) -> PermissionRequester:
            if name == "custom":
                return FakeUiRequester()
            raise Exception("Unexpected UI")

        monkeypatch.setattr(
            "little_harness.composition.discover_permission_requester", fake_discover
        )

        requester = build_permission_requester(config)
        assert isinstance(requester, FakeUiRequester)

    def test_falls_back_when_ui_requester_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = ArgumentParser().parse(["--prompt", "hi", "--ui", "custom"])
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)

        def fake_discover(name: str) -> PermissionRequester:
            raise UnknownPermissionRequesterError()

        monkeypatch.setattr(
            "little_harness.composition.discover_permission_requester", fake_discover
        )

        requester = build_permission_requester(config)
        assert isinstance(requester, InteractivePermissionRequester)


class TestCommandRegistryComposition:
    def test_builds_registry_with_built_ins(self) -> None:
        registry = build_command_registry()
        assert registry.get("/exit") is not None
        assert registry.get("/clear") is not None

    def test_builds_registry_including_plugins(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class FakePluginCommand:
            name = "plugged"
            aliases = ()
            description = "Plugin command"

            def execute(self, console: object, /) -> None:
                pass

        install_entry_points(
            monkeypatch,
            {
                REPL_COMMAND_GROUP: [
                    FakeEntryPoint("plugged_builder", FakePluginCommand)
                ]
            },
        )

        registry = build_command_registry()
        assert registry.get("/plugged") is not None
        assert registry.get("/exit") is not None


class _PrePopulatedRepo:
    @staticmethod
    def load(_sid: SessionId) -> MessageHistory:
        return MessageHistory().with_message(
            ChatMessage(USER, MessageContent("previous message"))
        )


class _PrePopulatedPlugin:
    @property
    def session_id(self) -> SessionId:
        return SessionId("test-session")

    def observer(self) -> AgentObserver:
        return RecordingObserver()

    def repository(self) -> _PrePopulatedRepo:
        return _PrePopulatedRepo()

    def fork(self) -> _PrePopulatedPlugin:
        return self


class _SpyingConsole:
    received_initial: ClassVar[list[object]] = []

    def __init__(self, *args: object, **kwargs: object) -> None:
        _SpyingConsole.received_initial.append(kwargs.get("_initial_messages"))

    @staticmethod
    def start() -> str:
        return ""


class _RecordingApp:
    calls: ClassVar[list[str]] = []

    def __enter__(self) -> _RecordingApp:
        return self

    def __exit__(self, *_: object) -> None:
        pass

    @staticmethod
    def build_system_message() -> object:
        return ChatMessage(SYSTEM, MessageContent("system"))

    @staticmethod
    def run_turn(
        prompt: Prompt, messages: MessageHistory
    ) -> tuple[object, MessageHistory]:
        _RecordingApp.calls.append("run_turn")
        return (
            AgentResult(MessageContent("resumed"), ElapsedSeconds(0.0), AgentSteps()),
            messages,
        )

    @staticmethod
    def run(_prompt: Prompt) -> str:
        _RecordingApp.calls.append("run")
        return "normal"


class TestSessionPluginWiring:
    """Session plugin discovery and wiring through the composition root."""

    def test_build_session_plugin_returns_none_when_no_plugin_installed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange: no session plugin entry points.
        install_entry_points(monkeypatch, {})
        config = ArgumentParser().parse(["--prompt", "hi"])

        plugin = _build_session_plugin(config)

        assert plugin is None

    def test_build_session_plugin_creates_plugin_in_interactive_mode(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange: install a fake session plugin builder.
        install_entry_points(
            monkeypatch,
            {
                POLICY_GROUP: [FakeEntryPoint("json", make_policy_builder())],
                SESSION_PLUGIN_GROUP: [
                    FakeEntryPoint(
                        "jsonl",
                        lambda policy, session_id=None: FakeSessionPlugin(  # type: ignore[reportUnknownLambdaType]
                            session_id=session_id  # type: ignore[reportUnknownArgumentType]
                        ),
                    )
                ],
            },
        )
        config = ArgumentParser().parse([])  # Interactive mode (no prompt)

        plugin = _build_session_plugin(config)

        assert isinstance(plugin, FakeSessionPlugin)

    def test_build_session_plugin_creates_plugin_when_session_id_given(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange: install a fake session plugin builder.
        install_entry_points(
            monkeypatch,
            {
                POLICY_GROUP: [FakeEntryPoint("json", make_policy_builder())],
                SESSION_PLUGIN_GROUP: [
                    FakeEntryPoint(
                        "jsonl",
                        lambda policy, session_id=None: FakeSessionPlugin(  # type: ignore[reportUnknownLambdaType]
                            session_id=session_id  # type: ignore[reportUnknownArgumentType]
                        ),
                    )
                ],
            },
        )
        config = ArgumentParser().parse(["--prompt", "hi", "--session", "my-session"])

        plugin = _build_session_plugin(config)

        assert isinstance(plugin, FakeSessionPlugin)
        assert plugin.session_id == SessionId("my-session")

    def test_run_cli_creates_session_plugin_in_interactive_mode(
        self,
        monkeypatch: pytest.MonkeyPatch,
        created_models: list[FakeChatModel],
    ) -> None:
        # Arrange: capture whether _build_session_plugin is called.
        called: list[bool] = []

        def fake_build_session_plugin(_config: AppConfig) -> SessionPlugin | None:
            called.append(True)
            return None

        def fake_start_console(_self: object) -> str:
            return ""

        monkeypatch.setattr(
            "little_harness.composition._build_session_plugin",
            fake_build_session_plugin,
        )
        monkeypatch.setattr(
            "little_harness.presentation.cli.interactive_console.InteractiveConsole.start",
            fake_start_console,
        )

        run_cli([])

        assert called == [True]

    def test_run_cli_returns_none_from_session_plugin_in_one_shot_without_session(
        self,
        monkeypatch: pytest.MonkeyPatch,
        created_models: list[FakeChatModel],
    ) -> None:
        # Arrange: spy on _build_session_plugin.
        returned: list[SessionPlugin | None] = []

        def fake_build_session_plugin(_config: AppConfig) -> SessionPlugin | None:
            result = None
            returned.append(result)
            return result

        monkeypatch.setattr(
            "little_harness.composition._build_session_plugin",
            fake_build_session_plugin,
        )

        run_cli(["--prompt", "hi"])

        assert returned == [None]

    def test_run_cli_wires_session_observer_when_plugin_built(
        self,
        monkeypatch: pytest.MonkeyPatch,
        created_models: list[FakeChatModel],
    ) -> None:
        # Arrange: a recognizable observer from the session plugin.
        sentinel = RecordingObserver()
        session_plugin = FakeSessionPlugin(observer=sentinel)
        received: list[AgentObserver] = []

        def fake_build_session_plugin(_config: AppConfig) -> SessionPlugin:
            return session_plugin

        def fake_build_application(
            _config: AppConfig,
            observer: AgentObserver,
            _skill_loader: object = None,
            _extra_hooks: object = None,
        ) -> FakeApplication:
            received.append(observer)
            return FakeApplication()

        def fake_start_console(_self: object) -> str:
            return ""

        monkeypatch.setattr(
            "little_harness.composition._build_session_plugin",
            fake_build_session_plugin,
        )
        monkeypatch.setattr(
            "little_harness.composition.build_application", fake_build_application
        )
        monkeypatch.setattr(
            "little_harness.presentation.cli.interactive_console.InteractiveConsole.start",
            fake_start_console,
        )

        run_cli([])

        assert received == [sentinel]

    def test_run_cli_passes_initial_messages_when_resuming(
        self,
        monkeypatch: pytest.MonkeyPatch,
        created_models: list[FakeChatModel],
    ) -> None:
        _SpyingConsole.received_initial = []

        monkeypatch.setattr(
            "little_harness.composition._build_session_plugin",
            lambda _config: _PrePopulatedPlugin(),  # type: ignore[reportUnknownLambdaType]
        )
        monkeypatch.setattr(
            "little_harness.composition.InteractiveConsole", _SpyingConsole
        )

        run_cli(["--session", "test-session"])

        assert len(_SpyingConsole.received_initial) == 1
        assert _SpyingConsole.received_initial[0] is not None

    def test_run_cli_one_shot_with_session_uses_run_turn(
        self,
        monkeypatch: pytest.MonkeyPatch,
        created_models: list[FakeChatModel],
    ) -> None:
        _RecordingApp.calls = []

        def fake_build_app(
            _config: object,
            _observer: object,
            _skill_loader: object = None,
            _extra_hooks: object = None,
        ) -> _RecordingApp:
            return _RecordingApp()

        monkeypatch.setattr(
            "little_harness.composition._build_session_plugin",
            lambda _config: FakeSessionPlugin(session_id=SessionId("test-session")),  # type: ignore[reportUnknownLambdaType]
        )
        monkeypatch.setattr(
            "little_harness.composition.build_application", fake_build_app
        )
        monkeypatch.setattr(
            "little_harness.composition.build_observer",
            lambda _config: None,  # type: ignore[reportUnknownLambdaType]
        )

        run_cli(["--prompt", "hi", "--session", "test-session"])

        assert _RecordingApp.calls == ["run_turn"]
