"""First-class collection of chat messages exchanged during an agent run."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from local_llm.domain.message import ChatMessage


@dataclass(frozen=True)
class MessageHistory:
    """An immutable, ordered conversation. Grow it with `with_message`.

    Example:
        history = MessageHistory().with_message(system_message)
    """

    _messages: tuple[ChatMessage, ...] = ()

    def with_message(self, message: ChatMessage) -> MessageHistory:
        return MessageHistory((*self._messages, message))

    def contains(self, message: ChatMessage) -> bool:
        return message in self._messages

    def __iter__(self) -> Iterator[ChatMessage]:
        return iter(self._messages)

    def __len__(self) -> int:
        return len(self._messages)
