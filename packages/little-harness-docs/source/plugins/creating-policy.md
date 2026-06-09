# Creating a policy plugin

An agent policy controls the system prompt format, response schema, and model output parsing. It implements the `AgentPolicy` port.

## Implement `AgentPolicy`

```python
from collections.abc import Sequence
from little_harness.application.ports.agent_policy import AgentPolicy
from little_harness.application.ports.chat_model import ResponseSchema
from little_harness.domain import (
    AgentDecision,
    ToolCall,
    FinalAnswer,
)
from little_harness.domain.message import ChatMessage
from little_harness.domain.tool_result import ToolRunResult
from little_harness.domain.tool_spec import ToolSpec
from little_harness.domain.values.role import ASSISTANT
from little_harness.domain.values.text_values import (
    MessageContent,
    Prompt,
    ToolInput,
    ToolName,
)


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
        import json
        parsed = json.loads(output)
        action = parsed.get("action", "")
        if action == "final":
            return FinalAnswer(MessageContent(parsed.get("answer", "")))
        return ToolCall(
            tool_name=ToolName(action),
            tool_input=ToolInput(json.dumps(parsed.get("action_input", {}))),
        )

    def build_tool_observation_message(
        self, prompt: Prompt, result: ToolRunResult
    ) -> ChatMessage:
        return ChatMessage(
            role=ASSISTANT,
            content=MessageContent(
                f"Tool {result.tool_name.value!r} "
                f"{'succeeded' if result.succeeded else 'failed'}: "
                f"{result.output.value}"
            ),
        )

    def build_repair_message(
        self, prompt: Prompt, error: Exception
    ) -> ChatMessage:
        return ChatMessage(
            role=ASSISTANT,
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

## The builder

```python
# src/little_harness_my_policy/provider.py
from little_harness.application.ports.agent_policy import AgentPolicy
from little_harness_my_policy.my_policy import MyPolicy


def build() -> AgentPolicy:
    return MyPolicy()
```

## Parsing details

The `parse_model_output` method receives the raw model output string and must return either a `ToolCall` or `FinalAnswer`. The built-in `JsonDecisionParser` in `little_harness_json_policy` shows the standard pattern:

1. Extract the first JSON object from the output (ignoring surrounding text/markdown)
2. Dispatch on the `action` field
3. Return the appropriate decision type

On failure, raise `AgentProtocolError` with a descriptive message — the runtime catches it and calls `build_repair_message` to ask the model to fix its output.
