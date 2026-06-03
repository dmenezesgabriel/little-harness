"""Factory selecting a `ChatModel` implementation by provider name.

Adding a provider (OpenAI, Ollama, ...) means registering a builder here; no
caller changes. Only llama.cpp exists today, so the builder takes its settings
directly — a future provider would generalize the settings argument.
"""

from __future__ import annotations

from collections.abc import Callable

from local_llm.application.ports.chat_model import ChatModel
from local_llm.infrastructure.llama_cpp.chat_model import LlamaCppChatModel
from local_llm.infrastructure.llama_cpp.settings import LlamaCppModelSettings

LLAMA_CPP_PROVIDER = "llama_cpp"

ChatModelBuilder = Callable[[LlamaCppModelSettings], ChatModel]


def build_llama_cpp_model(settings: LlamaCppModelSettings) -> ChatModel:
    return LlamaCppChatModel(settings)


CHAT_MODEL_BUILDERS: dict[str, ChatModelBuilder] = {
    LLAMA_CPP_PROVIDER: build_llama_cpp_model,
}


def create_chat_model(
    provider: str,
    settings: LlamaCppModelSettings,
) -> ChatModel:
    builder = CHAT_MODEL_BUILDERS.get(provider)

    if builder is None:
        known = sorted(CHAT_MODEL_BUILDERS)
        raise ValueError(f"Unknown provider: {provider!r}. Expected one of {known}.")

    return builder(settings)
