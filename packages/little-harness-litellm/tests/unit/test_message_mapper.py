from __future__ import annotations

from little_harness.domain.message import ChatMessage
from little_harness.domain.values.role import ASSISTANT, SYSTEM, USER
from little_harness.domain.values.text_values import MessageContent
from little_harness_litellm.message_mapper import to_litellm_message


class TestToLitellmMessage:
    def test_maps_role_name_and_content(self) -> None:
        # Arrange
        message = ChatMessage(SYSTEM, MessageContent("rules"))

        # Act / Assert
        assert to_litellm_message(message) == {"role": "system", "content": "rules"}

    def test_maps_each_role(self) -> None:
        # Act / Assert
        for role, name in (
            (SYSTEM, "system"),
            (USER, "user"),
            (ASSISTANT, "assistant"),
        ):
            message = ChatMessage(role, MessageContent("x"))
            assert to_litellm_message(message)["role"] == name
