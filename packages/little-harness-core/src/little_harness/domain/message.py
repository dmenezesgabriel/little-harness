"""A single chat message: a role paired with its content."""

from __future__ import annotations

from dataclasses import dataclass

from little_harness.domain.values.role import Role
from little_harness.domain.values.text_values import MessageContent


@dataclass(frozen=True)
class ChatMessage:
    """One message in a conversation.

    Example:
        message = ChatMessage(USER, MessageContent("What is 2 + 2?"))

    """

    role: Role
    content: MessageContent
