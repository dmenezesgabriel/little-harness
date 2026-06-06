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
```

## 2. Implement `ChatModel`

```python
# src/little_harness_my_provider/provider.py
from collections.abc import Iterator
from little_harness.application.ports import ChatModel
from little_harness.domain.values import MessageContent
from little_harness.domain import ChatCompletionRequest


class MyChatModel(ChatModel):
    def __init__(self, api_key: str, model: str) -> None:
        self._api_key = api_key
        self._model = model

    def complete_streaming(
        self, request: ChatCompletionRequest
    ) -> Iterator[MessageContent]:
        # Call your SDK, yield content chunks
        yield MessageContent("Hello from my provider!")

    def close(self) -> None:
        pass


def build(options: dict) -> ChatModel:
    return MyChatModel(
        api_key=options.get("api_key", ""),
        model=options["model"],
    )
```

## 3. Validate options

```python
# src/little_harness_my_provider/model_settings.py
from dataclasses import dataclass


@dataclass(frozen=True)
class MyModelSettings:
    model: str
    api_key: str = ""
    temperature: float | None = None

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
from little_harness_my_provider.provider import MyChatModel, build
from little_harness.domain import ChatCompletionRequest
from little_harness.domain.values import Role, MessageContent

def test_build_returns_chat_model() -> None:
    model = build({"model": "test-model"})
    assert isinstance(model, MyChatModel)

def test_complete_streaming_yields_content() -> None:
    model = MyChatModel(api_key="", model="test")
    request = ChatCompletionRequest(
        messages=[(Role.USER, MessageContent("hi"))],
    )
    chunks = list(model.complete_streaming(request))
    assert len(chunks) > 0
```
