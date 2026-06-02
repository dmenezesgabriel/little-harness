from __future__ import annotations

from local_llm.agent import AgentRuntime, AgentRuntimeConfig
from local_llm.calculator import CalculatorTool
from local_llm.cli import AppConfig, parse_args, print_result
from local_llm.json_policy import JsonAgentPolicy
from local_llm.llama_cpp_model import LlamaCppChatModel, LlamaCppModelSettings


def main() -> None:
    config = parse_args()
    agent = AgentRuntime(
        chat_model=create_chat_model(config),
        tools=[CalculatorTool()],
        policy=JsonAgentPolicy(),
        config=AgentRuntimeConfig(
            max_iterations=config.max_iterations,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        ),
    )

    result = agent.run(config.prompt)
    print_result(result)


def create_chat_model(config: AppConfig) -> LlamaCppChatModel:
    settings = LlamaCppModelSettings(
        model_path=config.model_path,
        context_size=config.context_size,
        thread_count=config.thread_count,
        gpu_layer_count=config.gpu_layer_count,
    )

    return LlamaCppChatModel(settings)


if __name__ == "__main__":
    main()
