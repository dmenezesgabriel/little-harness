from __future__ import annotations

from local_llm.domain.message import ChatMessage
from local_llm.domain.message_history import MessageHistory
from local_llm.domain.values.role import SYSTEM, USER
from local_llm.domain.values.text_values import MessageContent


def system_message() -> ChatMessage:
    return ChatMessage(SYSTEM, MessageContent("rules"))


def user_message() -> ChatMessage:
    return ChatMessage(USER, MessageContent("question"))


class TestMessageHistory:
    def test_with_message_appends_without_mutating_the_original(self) -> None:
        # Arrange
        empty = MessageHistory()

        # Act
        grown = empty.with_message(system_message()).with_message(user_message())

        # Assert
        assert len(empty) == 0
        assert list(grown) == [system_message(), user_message()]

    def test_contains_reports_membership(self) -> None:
        # Arrange
        history = MessageHistory().with_message(system_message())

        # Act / Assert
        assert history.contains(system_message()) is True
        assert history.contains(user_message()) is False
