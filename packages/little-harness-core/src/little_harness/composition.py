"""Composition root: the one place allowed to wire every layer together."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import replace

from little_harness.application.agent_dependencies import AgentDependencies
from little_harness.application.agent_runtime import AgentRuntime, AgentRuntimeConfig
from little_harness.application.hook_chain import HookChain
from little_harness.application.ports.agent_observer import AgentObserver
from little_harness.application.ports.agent_policy import AgentPolicy
from little_harness.application.ports.chat_model import ChatModel
from little_harness.application.ports.closeable import Closeable
from little_harness.application.ports.lifecycle_hook import LifecycleHook
from little_harness.application.ports.permission_requester import PermissionRequester
from little_harness.application.ports.skill_loader import SkillLoader
from little_harness.application.ports.token_sink import TokenSink
from little_harness.application.tool_registry import ToolRegistry
from little_harness.domain.errors import UnknownPermissionRequesterError
from little_harness.domain.message import ChatMessage
from little_harness.domain.message_history import MessageHistory
from little_harness.domain.result import AgentResult
from little_harness.domain.values.text_values import Prompt
from little_harness.domain.values.truncation import TruncationConfig
from little_harness.infrastructure.config.config_loader import ConfigLoader
from little_harness.infrastructure.hooks.approval_hook import ApprovalHook
from little_harness.infrastructure.observability.null_observer import NullObserver
from little_harness.infrastructure.permissions.auto_approve_requester import (
    AutoApprovePermissionRequester,
)
from little_harness.infrastructure.skills.file_system_skill_loader import (
    FileSystemSkillLoader,
)
from little_harness.infrastructure.truncation.head_truncator import HeadTruncator
from little_harness.plugin_discovery import (
    default_policy_name,
    default_provider_name,
    discover_observer,
    discover_permission_requester,
    discover_policy,
    discover_repl_commands,
    discover_tools,
    discover_ui,
    load_chat_model_builder,
)
from little_harness.presentation.cli.app_config import AppConfig
from little_harness.presentation.cli.argument_parser import ArgumentParser
from little_harness.presentation.cli.interactive_console import InteractiveConsole
from little_harness.presentation.cli.permission_prompt import (
    InteractivePermissionRequester,
)
from little_harness.presentation.cli.repl_command import (
    CommandRegistry,
    build_default_registry,
)
from little_harness.presentation.cli.result_renderer import ResultRenderer
from little_harness.presentation.cli.token_sinks import NullTokenSink, StdoutTokenSink


class Application:
    """Runs the agent for a prompt and renders the result as text.

    A context manager so the model's native resources are released on exit:
        with build_application(config) as app:
            print(app.run(prompt))
    """

    def __init__(
        self,
        runtime: AgentRuntime,
        renderer: ResultRenderer,
        chat_model: Closeable,
    ) -> None:
        """See class docstring for argument descriptions."""
        self._runtime = runtime
        self._renderer = renderer
        self._chat_model = chat_model

    def run(self, prompt: Prompt) -> str:
        """Run the agent with the given prompt and return rendered output."""
        return self._renderer.render(self._runtime.run(prompt))

    def build_system_message(self) -> ChatMessage:
        """Build the system message for the agent."""
        return self._runtime.build_system_message()

    def run_turn(
        self, prompt: Prompt, messages: MessageHistory
    ) -> tuple[AgentResult, MessageHistory]:
        """Run a single turn with the given prompt and message history."""
        return self._runtime.run_turn(prompt, messages)

    def __enter__(self) -> Application:
        """Enter the context manager, returning the application."""
        return self

    def __exit__(self, *_exc_info: object) -> None:
        """Close the chat model on exit."""
        self._chat_model.close()


def build_application(
    config: AppConfig,
    observer: AgentObserver | None = None,
    skill_loader: SkillLoader | None = None,
) -> Application:
    """Build and return an `Application` from config."""
    dependencies = build_dependencies(config, observer or NullObserver(), skill_loader)
    runtime = AgentRuntime(dependencies, to_runtime_config(config))
    return Application(runtime, ResultRenderer(), dependencies.chat_model)


def build_observer(config: AppConfig) -> AgentObserver:
    """Build an observer from config, or return `NullObserver`."""
    # No observer selected means no observability; only a named plugin is loaded.
    if config.observer_name is None:
        return NullObserver()

    return discover_observer(config.observer_name)


def build_dependencies(
    config: AppConfig,
    observer: AgentObserver,
    skill_loader: SkillLoader | None = None,
) -> AgentDependencies:
    """Build all agent dependencies from config and observer."""
    registry = ToolRegistry(discover_tools(config.tool_selection))
    return AgentDependencies(
        chat_model=build_chat_model(config),
        tool_registry=registry,
        policy=build_policy(config),
        observer=observer,
        token_sink=build_token_sink(config),
        hooks=build_hooks(registry, config),
        truncator=HeadTruncator(),
        truncation_config=TruncationConfig(),
        skill_loader=skill_loader or FileSystemSkillLoader(config.skill_paths),
    )


def build_hooks(registry: ToolRegistry, config: AppConfig) -> LifecycleHook:
    """Build the lifecycle hook chain from config."""
    # The seam: every lifecycle hook is composed here. An empty chain folds to
    # `Proceed` (like the null hook), so adding a second hook needs no rewiring.
    return HookChain(build_hook_list(registry, config))


def build_hook_list(registry: ToolRegistry, config: AppConfig) -> list[LifecycleHook]:
    """Build the list of lifecycle hooks from config."""
    names_requiring_approval = approval_required_names(registry)

    if not names_requiring_approval:
        return []

    return [ApprovalHook(build_permission_requester(config), names_requiring_approval)]


def approval_required_names(registry: ToolRegistry) -> frozenset[str]:
    """Return the set of tool names that require approval."""
    return frozenset(
        spec.name.value for spec in registry.specs() if spec.requires_approval
    )


def build_permission_requester(config: AppConfig) -> PermissionRequester:
    """Build the permission requester from config."""
    # A terminal is required to prompt a human; piped input, CI, and `--yes` all
    # run unattended, so they auto-approve and rely on each tool's guardrails.
    if config.approve_all or not sys.stdin.isatty():
        return AutoApprovePermissionRequester()

    if config.ui != "default":
        try:
            return discover_permission_requester(config.ui)
        except UnknownPermissionRequesterError:
            pass

    return InteractivePermissionRequester()


def build_chat_model(config: AppConfig) -> ChatModel:
    """Build the chat model from config."""
    provider = config.provider or default_provider_name()
    builder = load_chat_model_builder(provider)
    # Plugin config from TOML is the floor; CLI --option/-o overrides it.
    merged_options = dict(config.plugin_configs.get(provider, {}))
    merged_options.update(config.provider_options)
    return builder(merged_options)


def build_policy(config: AppConfig) -> AgentPolicy:
    """Build the agent policy from config."""
    # Discovery imports only the selected policy's adapter; core ships none, so an
    # omitted --policy resolves to the sole installed policy.
    policy = config.policy or default_policy_name()
    return discover_policy(policy)


def build_token_sink(config: AppConfig) -> TokenSink:
    """Build the token sink from config."""
    if not config.enable_streaming:
        return NullTokenSink()

    return StdoutTokenSink()


def build_command_registry() -> CommandRegistry:
    """Build the command registry from built-ins and installed plugins."""
    registry = build_default_registry()

    for command in discover_repl_commands():
        registry.add(command, f"plugin:{type(command).__module__}")

    return registry


def to_runtime_config(config: AppConfig) -> AgentRuntimeConfig:
    """Convert `AppConfig` to `AgentRuntimeConfig`."""
    return AgentRuntimeConfig(
        max_iterations=config.max_iterations,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        top_p=config.top_p,
        repeat_penalty=config.repeat_penalty,
    )


def run_cli(argv: Sequence[str] | None = None) -> str:
    """Parse CLI args, load TOML config, resolve profile, and run the application."""
    argv_seq = list(argv) if argv is not None else []

    profile_cli = _extract_profile(argv_seq)
    loader = _create_config_loader()
    toml_config = loader.load()

    profile_name = profile_cli or toml_config.profile
    if profile_name:
        toml_config = loader.resolve_profile(toml_config, profile_name)

    app_config = ArgumentParser(toml_config).parse(argv_seq)
    app_config = replace(app_config, profile=profile_name)

    skill_loader = FileSystemSkillLoader(app_config.skill_paths)

    with build_application(app_config, build_observer(app_config), skill_loader) as app:
        if app_config.prompt is None:
            registry = build_command_registry()
            if app_config.ui == "default":
                return InteractiveConsole(
                    app, registry=registry, skill_loader=skill_loader
                ).start()
            ui_builder = discover_ui(app_config.ui)
            return ui_builder(app, registry).start()
        return app.run(app_config.prompt)


def _create_config_loader() -> ConfigLoader:
    """Factory for the real ConfigLoader. Easy to monkeypatch in tests."""
    return ConfigLoader()


def _extract_profile(argv: Sequence[str]) -> str | None:
    """Pre-parse ``--profile`` from argv so config can use it for profile resolution."""
    if not argv:
        return None
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--profile", default=None)
    known, _remaining = pre.parse_known_args(argv)
    return known.profile
