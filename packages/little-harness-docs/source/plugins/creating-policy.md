# Creating a policy plugin

An agent policy controls the system prompt format, response schema, and model
output parsing. It implements the `AgentPolicy` port.

## Implement `AgentPolicy`

```python
from collections.abc import Sequence
from little_harness.application.ports import AgentPolicy
from little_harness.domain import (
    AgentDecision,
    ToolSpec,
    ChatMessage,
    ResponseSchema,
)
from little_harness.domain.values import MessageContent


class MyPolicy(AgentPolicy):
    def system_prompt(self, tools: Sequence[ToolSpec]) -> MessageContent:
        return MessageContent(
            "You are a helpful assistant. "
            "Output your response as JSON."
        )

    def response_schema(self, tools: Sequence[ToolSpec]) -> ResponseSchema:
        return ResponseSchema({
            "type": "object",
            "properties": {
                "thought": {"type": "string"},
                "action": {"type": "string"},
                "action_input": {},
            },
        })

    def parse_model_output(self, output: str) -> AgentDecision:
        # Parse JSON output, return ToolCall or FinalAnswer
        ...

    def build_tool_observation_message(
        self, tool_name: str, output: str, success: bool
    ) -> ChatMessage:
        return ChatMessage(
            role=...,
            content=MessageContent(f"Tool {tool_name} returned: {output}"),
        )

    def build_repair_message(
        self, raw: str, error: str
    ) -> ChatMessage:
        return ChatMessage(
            role=...,
            content=MessageContent(
                f"Parse error: {error}. Please fix your output."
            ),
        )
```

## Register the entry point

```toml
[project.entry-points."little_harness.agent_policies"]
my_policy = "little_harness_my_policy.provider:build"
```
