"""Composition root: the one place allowed to wire every layer together."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from local_llm.application.agent_dependencies import AgentDependencies
from local_llm.application.agent_runtime import AgentRuntime, AgentRuntimeConfig
from local_llm.application.ports.agent_observer import AgentObserver
from local_llm.application.ports.chat_model import ChatModel
from local_llm.application.ports.closeable import Closeable
from local_llm.application.ports.lifecycle_hook import LifecycleHook
from local_llm.application.ports.token_sink import TokenSink
from local_llm.application.tool_registry import ToolRegistry
from local_llm.domain.values.text_values import Prompt
from local_llm.infrastructure.hooks.null_hook import NullHook
from local_llm.infrastructure.llama_cpp.settings import LlamaCppModelSettings
from local_llm.infrastructure.observability.null_observer import NullObserver
from local_llm.infrastructure.observability.stdlib_logger import (
    create_structured_logger,
)
from local_llm.infrastructure.observability.structured_logging_observer import (
    StructuredLoggingObserver,
)
from local_llm.infrastructure.policy.json_agent_policy import JsonAgentPolicy
from local_llm.infrastructure.providers.chat_model_factory import build_llama_cpp_model
from local_llm.infrastructure.tools.calculator.calculator_tool import CalculatorTool
from local_llm.presentation.cli.app_config import AppConfig
from local_llm.presentation.cli.argument_parser import ArgumentParser
from local_llm.presentation.cli.result_renderer import ResultRenderer
from local_llm.presentation.cli.token_sinks import NullTokenSink, StdoutTokenSink

LOGGER_NAME = "agent"
LLAMA_CPP_PROVIDER = "llama_cpp"

ChatModelFromConfig = Callable[[AppConfig], ChatModel]


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
        tool_registry=ToolRegistry([CalculatorTool()]),
        policy=JsonAgentPolicy(),
        observer=observer,
        token_sink=build_token_sink(config),
        hooks=build_hooks(),
    )


def build_hooks() -> LifecycleHook:
    # The seam: wrap real hooks in `HookChain([...])` here, the one place tools
    # are registered too. No hooks are configured by default.
    return NullHook()


def build_llama_cpp_chat_model(config: AppConfig) -> ChatModel:
    return build_llama_cpp_model(to_llama_settings(config))


CHAT_MODEL_BUILDERS: dict[str, ChatModelFromConfig] = {
    LLAMA_CPP_PROVIDER: build_llama_cpp_chat_model,
}


def build_chat_model(config: AppConfig) -> ChatModel:
    builder = CHAT_MODEL_BUILDERS.get(config.provider)

    if builder is None:
        known = sorted(CHAT_MODEL_BUILDERS)
        raise ValueError(
            f"Unknown provider: {config.provider!r}. Expected one of {known}."
        )

    return builder(config)


def build_token_sink(config: AppConfig) -> TokenSink:
    if not config.enable_streaming:
        return NullTokenSink()

    return StdoutTokenSink()


def to_llama_settings(config: AppConfig) -> LlamaCppModelSettings:
    return LlamaCppModelSettings(
        model_path=config.model_path,
        context_size=config.context_size,
        thread_count=config.thread_count,
        gpu_layer_count=config.gpu_layer_count,
    )


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
