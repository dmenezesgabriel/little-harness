"""Composition root: the one place allowed to wire every layer together."""

from __future__ import annotations

from collections.abc import Sequence

from local_llm.application.agent_dependencies import AgentDependencies
from local_llm.application.agent_runtime import AgentRuntime, AgentRuntimeConfig
from local_llm.application.ports.agent_observer import AgentObserver
from local_llm.application.tool_registry import ToolRegistry
from local_llm.domain.values.text_values import Prompt
from local_llm.infrastructure.llama_cpp.settings import LlamaCppModelSettings
from local_llm.infrastructure.observability.null_observer import NullObserver
from local_llm.infrastructure.policy.json_agent_policy import JsonAgentPolicy
from local_llm.infrastructure.providers.chat_model_factory import (
    LLAMA_CPP_PROVIDER,
    create_chat_model,
)
from local_llm.infrastructure.tools.calculator.calculator_tool import CalculatorTool
from local_llm.presentation.cli.app_config import AppConfig
from local_llm.presentation.cli.argument_parser import ArgumentParser
from local_llm.presentation.cli.result_renderer import ResultRenderer


class Application:
    """Runs the agent for a prompt and renders the result as text."""

    def __init__(self, runtime: AgentRuntime, renderer: ResultRenderer) -> None:
        self._runtime = runtime
        self._renderer = renderer

    def run(self, prompt: Prompt) -> str:
        return self._renderer.render(self._runtime.run(prompt))


def build_application(
    config: AppConfig,
    observer: AgentObserver | None = None,
) -> Application:
    dependencies = build_dependencies(config, observer or NullObserver())
    runtime = AgentRuntime(dependencies, to_runtime_config(config))
    return Application(runtime, ResultRenderer())


def build_dependencies(
    config: AppConfig,
    observer: AgentObserver,
) -> AgentDependencies:
    chat_model = create_chat_model(LLAMA_CPP_PROVIDER, to_llama_settings(config))
    return AgentDependencies(
        chat_model=chat_model,
        tool_registry=ToolRegistry([CalculatorTool()]),
        policy=JsonAgentPolicy(),
        observer=observer,
    )


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
    return build_application(config).run(config.prompt)
