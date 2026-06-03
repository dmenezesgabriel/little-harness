"""Maps domain chat messages to llama.cpp messages via per-role builders.

Keying builders by `Role` replaces the original `if role == ...` chain: each
role resolves its own message constructor.
"""

from __future__ import annotations

from collections.abc import Callable

from llama_cpp.llama_types import (
    ChatCompletionRequestAssistantMessage,
    ChatCompletionRequestMessage,
    ChatCompletionRequestSystemMessage,
    ChatCompletionRequestUserMessage,
)

from local_llm.domain.message import ChatMessage
from local_llm.domain.values.role import ASSISTANT, SYSTEM, USER, Role

RoleMessageBuilder = Callable[[str], ChatCompletionRequestMessage]


def to_llama_message(message: ChatMessage) -> ChatCompletionRequestMessage:
    builder = ROLE_MESSAGE_BUILDERS[message.role]
    return builder(message.content.value)


def build_system_message(content: str) -> ChatCompletionRequestMessage:
    return ChatCompletionRequestSystemMessage(role="system", content=content)


def build_user_message(content: str) -> ChatCompletionRequestMessage:
    return ChatCompletionRequestUserMessage(role="user", content=content)


def build_assistant_message(content: str) -> ChatCompletionRequestMessage:
    return ChatCompletionRequestAssistantMessage(role="assistant", content=content)


ROLE_MESSAGE_BUILDERS: dict[Role, RoleMessageBuilder] = {
    SYSTEM: build_system_message,
    USER: build_user_message,
    ASSISTANT: build_assistant_message,
}
