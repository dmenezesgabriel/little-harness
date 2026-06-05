from __future__ import annotations

from little_harness.application.ports.agent_policy import AgentPolicy
from little_harness.domain.decision import FinalAnswer
from little_harness.domain.values.text_values import MessageContent
from little_harness_json_policy.provider import build


class TestBuild:
    def test_returns_a_policy_conforming_to_the_port(self) -> None:
        # Arrange: the annotation forces a structural-conformance check.
        policy: AgentPolicy = build()

        # Act / Assert: the built policy parses the strict-JSON protocol.
        output = MessageContent(
            '{"action":"final","tool_name":null,"tool_input":null,"answer":"hi"}'
        )
        assert policy.parse_model_output(output) == FinalAnswer(MessageContent("hi"))
