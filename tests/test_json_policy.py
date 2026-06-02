from __future__ import annotations

import pytest

from local_llm.agent import AgentProtocolError
from local_llm.json_policy import JsonAgentPolicy


class TestJsonAgentPolicyParseModelOutput:
    def test_accepts_final_json_with_surrounding_text(self) -> None:
        # Arrange
        policy = JsonAgentPolicy()
        output = (
            'ignored prefix {"action":"final","tool_name":null,'
            '"tool_input":null,"answer":" done "}'
        )

        # Act
        decision = policy.parse_model_output(output)

        # Assert
        assert decision.kind == "final"
        assert decision.final_answer == "done"

    def test_rejects_missing_json_object(self) -> None:
        # Arrange
        policy = JsonAgentPolicy()
        output = "plain text"

        # Act / Assert
        with pytest.raises(AgentProtocolError, match="Could not find JSON object"):
            policy.parse_model_output(output)

    @pytest.mark.parametrize(
        ("payload", "message"),
        [
            (
                '{"action":"invalid","tool_name":null,"tool_input":null,"answer":null}',
                "Expected action",
            ),
            (
                '{"action":"final","tool_name":null,"tool_input":null,"answer":null}',
                "Expected non-null string",
            ),
            (
                '{"action":"tool","tool_name":null,"tool_input":"2 + 2","answer":null}',
                "Expected non-null string",
            ),
            (
                '{"action":"tool","tool_name":"calculator",'
                '"tool_input":null,"answer":null}',
                "Expected non-null string",
            ),
            (
                '{"action":"final","tool_name":1,"tool_input":null,"answer":"done"}',
                "Expected tool_name",
            ),
        ],
    )
    def test_rejects_invalid_decision_shape(
        self,
        payload: str,
        message: str,
    ) -> None:
        # Arrange
        policy = JsonAgentPolicy()

        # Act / Assert
        with pytest.raises(AgentProtocolError, match=message):
            policy.parse_model_output(payload)
