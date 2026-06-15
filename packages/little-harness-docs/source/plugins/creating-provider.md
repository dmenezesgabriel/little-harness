# Creating a provider plugin

A provider plugin wraps a chat-model SDK behind the `ChatModel` port.

## 1. Create the package

```
little-harness-my-provider/
  pyproject.toml
  src/
    little_harness_my_provider/
      __init__.py
      provider.py
      model_settings.py
      chat_model.py
```

## 2. Implement `ChatModel`

```python
# src/little_harness_my_provider/chat_model.py
from collections.abc import Iterator
from little_harness.application.ports.chat_model import ChatModel, ChatCompletionRequest
from little_harness.domain.values.text_values import MessageContent


class MyChatModel(ChatModel):
    def __init__(self, api_key: str, model: str) -> None:
        self._api_key = api_key
        self._model = model

    def supports_thinking(self) -> bool:
        """Return True when the provider supports reasoning tokens."""
        return True

    def complete_streaming(
        self, request: ChatCompletionRequest
    ) -> Iterator[MessageContent]:
        # Call your SDK, yield content chunks
        yield MessageContent("Hello from my provider!")

    def close(self) -> None:
        """Release provider resources (e.g. close HTTP sessions)."""
        pass
```

### ChatCompletionRequest structure

`ChatCompletionRequest` carries everything needed for a model call:

| Field | Type | Description |
|-------|------|-------------|
| `messages` | `Sequence[ChatMessage]` | Full conversation history |
| `temperature` | `Temperature` | Sampling temperature |
| `max_tokens` | `MaxTokens` | Max tokens to generate |
| `response_schema` | `ResponseSchema \| None` | JSON Schema for constrained decoding |
| `top_p` | `TopP \| None` | Nucleus sampling threshold |
| `repeat_penalty` | `RepeatPenalty \| None` | Repetition penalty |
| `thinking_level` | `ThinkingLevel \| None` | How much reasoning to expose (off, low, medium, high) |
| `thinking_budget` | `ThinkingBudget \| None` | Max tokens allowed for reasoning |

Map these to your SDK's native request format in `complete_streaming`. If your
provider does not support thinking, ignore the two thinking fields — the runtime
will leave them `None` for non-thinking models.

### `supports_thinking()`

Override `supports_thinking()` to return `True` when the model provider supports
reasoning/thinking tokens. The default returns `False`, so providers that do not
support thinking can omit the override entirely.

### Streaming

`complete_streaming` must return an `Iterator[MessageContent]` — each chunk is a
fragment of the full response. The runtime concatenates all chunks and emits each
one through the `TokenSink` for live UI output. For non-streaming SDKs, yield a
single chunk:

```python
def complete_streaming(self, request: ChatCompletionRequest) -> Iterator[MessageContent]:
    response = self._sdk.complete(request.messages)  # blocking call
    yield MessageContent(response.text)
```

When the model produces both reasoning and visible tokens, yield chunks with
`thinking` set for the reasoning portion:

```python
from little_harness.domain.values.thinking import ThinkingContent

def complete_streaming(self, request: ChatCompletionRequest) -> Iterator[MessageContent]:
    for chunk in self._sdk.complete_streaming(request.messages):
        if chunk.type == "reasoning":
            yield MessageContent("", thinking=ThinkingContent(chunk.text))
        else:
            yield MessageContent(chunk.text)
```

### Response Schema / Constrained Decoding

If the provider supports JSON Schema-guided generation (e.g. OpenAI's
`response_format`, llama.cpp's `grammar`), convert the `ResponseSchema` value to
your SDK's format in a helper method. If not supported, ignore it — the policy
will still work through prompting alone.

```python
def to_response_format(self, schema: ResponseSchema | None) -> dict | None:
    if schema is None:
        return None
    return {"type": "json_object", "schema": schema.value}
```

## 3. Validate options

```python
# src/little_harness_my_provider/model_settings.py
from dataclasses import dataclass


@dataclass(frozen=True)
class MyModelSettings:
    model: str
    api_key: str = ""
    num_retries: int = 0

    @classmethod
    def from_options(cls, options: dict) -> "MyModelSettings":
        if "model" not in options:
            raise ValueError("'model' is required")
        return cls(
            model=options["model"],
            api_key=options.get("api_key", ""),
        )
```

## 4. Register the entry point

```toml
# pyproject.toml
[project]
name = "little-harness-my-provider"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = ["little-harness-core"]

[project.entry-points."little_harness.chat_model_providers"]
my_provider = "little_harness_my_provider.provider:build"
```

## 5. Add to the umbrella extra

In `little-harness/pyproject.toml`:

```toml
[project.optional-dependencies]
my-provider = ["little-harness-my-provider"]
all = [
    "little-harness-llama-cpp",
    # ... other extras
    "little-harness-my-provider",
]
```

Users select it with:

```bash
little-harness --provider my_provider --model my-model -p "Hello!"
```

## 6. Write tests

```python
from little_harness_my_provider import MyChatModel, build
from little_harness.application.ports.chat_model import ChatCompletionRequest
from little_harness.domain.message import ChatMessage
from little_harness.domain.message_history import MessageHistory
from little_harness.domain.values.role import USER
from little_harness.domain.values.text_values import MessageContent


def test_build_returns_chat_model() -> None:
    model = build({"model": "test-model"})
    assert isinstance(model, MyChatModel)


def test_complete_streaming_yields_content() -> None:
    model = MyChatModel(api_key="", model="test")
    history = MessageHistory().with_message(
        ChatMessage(USER, MessageContent("hi"))
    )
    request = ChatCompletionRequest(
        messages=history,
        temperature=...,
        max_tokens=...,
        response_schema=None,
    )
    chunks = list(model.complete_streaming(request))
    assert len(chunks) > 0
```
