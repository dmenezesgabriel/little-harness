"""Builders that construct a `ChatModel` from a provider's own settings.

Each provider package owns one builder. Provider *selection* by name lives in the
composition root (it depends on `AppConfig`, which infrastructure may not import),
so adding a provider is: a new adapter package + one builder here (or there) + one
registry entry in `composition.py`. No application/domain change.
"""

from __future__ import annotations

from local_llm.application.ports.chat_model import ChatModel
from local_llm.infrastructure.llama_cpp.chat_model import LlamaCppChatModel
from local_llm.infrastructure.llama_cpp.settings import LlamaCppModelSettings


def build_llama_cpp_model(settings: LlamaCppModelSettings) -> ChatModel:
    return LlamaCppChatModel(settings)
