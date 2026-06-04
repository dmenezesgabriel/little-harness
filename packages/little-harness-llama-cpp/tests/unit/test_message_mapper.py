from __future__ import annotations

import pytest
from little_harness.domain.message import ChatMessage
from little_harness.domain.values.role import ASSISTANT, SYSTEM, USER, Role
from little_harness.domain.values.text_values import MessageContent
from little_harness_llama_cpp.message_mapper import to_llama_message


class TestToLlamaMessage:
    @pytest.mark.parametrize(
        ("role", "expected_role"),
        [(SYSTEM, "system"), (ASSISTANT, "assistant"), (USER, "user")],
    )
    def test_maps_each_role_to_its_llama_message(
        self,
        role: Role,
        expected_role: str,
    ) -> None:
        # Arrange
        message = ChatMessage(role, MessageContent("text"))

        # Act
        llama_message = to_llama_message(message)

        # Assert
        assert llama_message["role"] == expected_role
        assert llama_message.get("content") == "text"
