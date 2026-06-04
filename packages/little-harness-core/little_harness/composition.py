"""Composition root: the one place allowed to wire every layer together."""

from __future__ import annotations

from collections.abc import Sequence

from little_harness.application.agent_dependencies import AgentDependencies
from little_harness.application.agent_runtime import AgentRuntime, AgentRuntimeConfig
from little_harness.application.ports.agent_observer import AgentObserver
from little_harness.application.ports.chat_model import ChatModel
from little_harness.application.ports.closeable import Closeable
from little_harness.application.ports.lifecycle_hook import LifecycleHook
from little_harness.application.ports.token_sink import TokenSink
from little_harness.application.tool_registry import ToolRegistry
from little_harness.domain.values.text_values import Prompt
from little_harness.infrastructure.hooks.null_hook import NullHook
from little_harness.infrastructure.observability.null_observer import NullObserver
from little_harness.infrastructure.observability.stdlib_logger import (
    create_structured_logger,
)
from little_harness.infrastructure.observability.structured_logging_observer import (
    StructuredLoggingObserver,
)
from little_harness.infrastructure.policy.json_agent_policy import JsonAgentPolicy
from little_harness.plugin_discovery import discover_tools, load_chat_model_builder
from little_harness.presentation.cli.app_config import AppConfig
from little_harness.presentation.cli.argument_parser import ArgumentParser
from little_harness.presentation.cli.result_renderer import ResultRenderer
from little_harness.presentation.cli.token_sinks import NullTokenSink, StdoutTokenSink

LOGGER_NAME = "agent"


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
        self._runtime = runtime
        self._renderer = renderer
        self._chat_model = chat_model

    def run(self, prompt: Prompt) -> str:
        return self._renderer.render(self._runtime.run(prompt))

    def __enter__(self) -> Application:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self._chat_model.close()


def build_application(
    config: AppConfig,
    observer: AgentObserver | None = None,
) -> Application:
    dependencies = build_dependencies(config, observer or NullObserver())
    runtime = AgentRuntime(dependencies, to_runtime_config(config))
    return Application(runtime, ResultRenderer(), dependencies.chat_model)


def build_observer(config: AppConfig) -> AgentObserver:
    if not config.enable_logging:
        return NullObserver()

    return StructuredLoggingObserver(create_structured_logger(LOGGER_NAME))


def build_dependencies(
    config: AppConfig,
    observer: AgentObserver,
) -> AgentDependencies:
    return AgentDependencies(
        chat_model=build_chat_model(config),
        tool_registry=ToolRegistry(discover_tools()),
        policy=JsonAgentPolicy(),
        observer=observer,
        token_sink=build_token_sink(config),
        hooks=build_hooks(),
    )


def build_hooks() -> LifecycleHook:
    # The seam: wrap real hooks in `HookChain([...])` here. None by default.
    return NullHook()


def build_chat_model(config: AppConfig) -> ChatModel:
    # Discovery imports only the selected provider's adapter (and its vendor SDK).
    builder = load_chat_model_builder(config.provider)
    return builder(config.provider_options)


def build_token_sink(config: AppConfig) -> TokenSink:
    if not config.enable_streaming:
        return NullTokenSink()

    return StdoutTokenSink()


def to_runtime_config(config: AppConfig) -> AgentRuntimeConfig:
    return AgentRuntimeConfig(
        max_iterations=config.max_iterations,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )


def run_cli(argv: Sequence[str] | None = None) -> str:
    config = ArgumentParser().parse(argv)
    with build_application(config, build_observer(config)) as app:
        return app.run(config.prompt)
