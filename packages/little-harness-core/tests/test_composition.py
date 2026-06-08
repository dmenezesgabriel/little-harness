from __future__ import annotations

from collections.abc import Mapping

import pytest
from little_harness.application.agent_runtime import AgentRuntimeConfig
from little_harness.application.hook_chain import HookChain
from little_harness.application.ports.agent_observer import AgentObserver
from little_harness.application.ports.agent_tool import AgentTool
from little_harness.application.ports.chat_model import ChatModel
from little_harness.application.ports.lifecycle_hook import LifecycleHook
from little_harness.application.tool_registry import ToolRegistry
from little_harness.composition import (
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
from little_harness.domain.errors import UnknownPolicyError, UnknownProviderError
from little_harness.domain.hook_decision import Proceed
from little_harness.domain.message_history import MessageHistory
from little_harness.domain.tool_result import ToolRunRequest, ToolRunResult
from little_harness.domain.tool_spec import ToolInputSchema, ToolSpec
from little_harness.domain.values.numeric_values import (
    Iteration,
    MaxIterations,
    MaxTokens,
    RepeatPenalty,
    Temperature,
    TopP,
)
from little_harness.domain.values.role import SYSTEM
from little_harness.domain.values.text_values import (
    MessageContent,
    Prompt,
    RunId,
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

        def fake_build_hooks(_registry: ToolRegistry, config: AppConfig) -> HookChain:
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
            _registry: ToolRegistry, config: AppConfig
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
            _config: AppConfig, observer: AgentObserver
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

        monkeypatch.setattr(
            "little_harness.presentation.cli.interactive_console.InteractiveConsole.start",
            fake_start,
        )
        monkeypatch.setattr(
            "little_harness.composition.build_application",
            lambda _config, _observer: FakeApplication(),
        )
        monkeypatch.setattr(
            "little_harness.composition.build_observer", lambda _config: None
        )

        result = run_cli([])

        assert result == ""
        assert started == [True]

    def test_invokes_build_application_when_no_prompt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        built: list[object] = []

        def fake_build(_config: AppConfig, _observer: AgentObserver) -> FakeApplication:
            built.append(_config)
            return FakeApplication()

        monkeypatch.setattr(
            "little_harness.presentation.cli.interactive_console.InteractiveConsole.start",
            lambda _self: "",
        )
        monkeypatch.setattr("little_harness.composition.build_application", fake_build)
        monkeypatch.setattr(
            "little_harness.composition.build_observer", lambda _config: None
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

        monkeypatch.setattr(
            "little_harness.presentation.cli.interactive_console.InteractiveConsole.start",
            lambda _self: "",
        )
        monkeypatch.setattr(
            "little_harness.composition.build_application",
            lambda _config, _observer: FakeApplication(),
        )
        monkeypatch.setattr("little_harness.composition.build_observer", fake_build)

        run_cli([])

        assert len(built) == 1

    def test_passes_built_application_to_interactive_console(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class SpyingInteractiveConsole:
            received_app: object = None

            def __init__(
                self,
                app: object,
                output: object = None,
                source: object = None,
                registry: object = None,
            ) -> None:
                SpyingInteractiveConsole.received_app = app

            def start(self) -> str:
                return ""

        built_app = FakeApplication()
        monkeypatch.setattr(
            "little_harness.composition.build_application",
            lambda _config, _observer: built_app,
        )
        monkeypatch.setattr(
            "little_harness.composition.build_observer", lambda _config: None
        )
        monkeypatch.setattr(
            "little_harness.composition.InteractiveConsole",
            SpyingInteractiveConsole,
        )

        run_cli([])

        assert SpyingInteractiveConsole.received_app is built_app

    def test_starts_custom_ui_when_selected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        passed_args: list[tuple[object, object]] = []

        class FakeCustomUi:
            def __init__(self, app: object, registry: object) -> None:
                passed_args.append((app, registry))

            def start(self) -> str:
                return "custom_started"

        built_app = FakeApplication()
        monkeypatch.setattr(
            "little_harness.composition.build_application",
            lambda _config, _observer: built_app,
        )
        monkeypatch.setattr(
            "little_harness.composition.build_observer", lambda _config: None
        )
        monkeypatch.setattr(
            "little_harness.composition.discover_ui",
            lambda name: FakeCustomUi if name == "custom" else None,
        )

        result = run_cli(["--ui", "custom"])

        assert result == "custom_started"
        assert len(passed_args) == 1
        assert passed_args[0][0] is built_app


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

        # Act / Assert
        assert isinstance(
            build_permission_requester(config), InteractivePermissionRequester
        )


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
