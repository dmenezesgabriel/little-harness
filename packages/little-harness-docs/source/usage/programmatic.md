# Programmatic usage

## The composition root

The simplest way to use `little-harness` from code is through
`run_cli` — it parses a `sys.argv`-style list and returns rendered text:

```python
from little_harness.composition import run_cli

text = run_cli(["--prompt", "Hello!"])
print(text)
```

## Build the full application

For more control, use `build_application` which wires every layer:

```python
from little_harness.composition import build_application, AppConfig
from little_harness.domain.values.numeric_values import Temperature, MaxTokens, MaxIterations
from little_harness.domain.values.text_values import Prompt

config = AppConfig(
    provider="litellm",
    provider_options={"model": "gemini/gemini-2.5-flash"},
)

with build_application(config) as app:
    result = app.run(Prompt("What is 144 divided by 12?"))
    print(result)
```

`Application` is a context manager that releases the model's resources on exit.

## Running multi-turn interactions

Use `run_turn` to run one iteration of the agent loop, returning the updated
message history for the next turn:

```python
from little_harness.domain.message_history import MessageHistory
from little_harness.domain.values.text_values import Prompt

with build_application(config) as app:
    system = app.build_system_message()
    messages = MessageHistory().with_message(system)

    result, messages = app.run_turn(Prompt("What is 144 / 12?"), messages)
    print(result.answer)

    result, messages = app.run_turn(Prompt("Now tell me if it's even."), messages)
    print(result.answer)
```

## The runtime directly

For full control, wire the `AgentRuntime` yourself:

```python
from little_harness.application import AgentRuntime, AgentDependencies, AgentRuntimeConfig, ToolRegistry
from little_harness.domain.values.numeric_values import Temperature, MaxTokens, MaxIterations
from little_harness.plugin_discovery import (
    load_chat_model_builder,
    discover_policy,
    discover_tools,
)

model = load_chat_model_builder("litellm")({"model": "gemini/gemini-2.5-flash"})
policy = discover_policy("json")
tools = ToolRegistry(discover_tools())

deps = AgentDependencies(
    model=model,
    policy=policy,
    tools=tools,
    observer=None,
    token_sink=None,
    hooks=None,
)

config = AgentRuntimeConfig(
    max_iterations=MaxIterations(5),
    temperature=Temperature(0.0),
    max_tokens=MaxTokens(512),
)

runtime = AgentRuntime(deps, config)
result = runtime.run(Prompt("What is 144 divided by 12?"))

print(f"Answer: {result.answer.value}")
print(f"Elapsed: {result.elapsed.value:.2f}s")
print(f"Steps: {len(result.steps)}")
```

### AgentDependencies fields

| Field | Type | Description |
|-------|------|-------------|
| `chat_model` | `ChatModel` | The model adapter |
| `tool_registry` | `ToolRegistry` | Registered tools |
| `policy` | `AgentPolicy` | The agent protocol policy |
| `observer` | `AgentObserver \| None` | Lifecycle observer (no-op when None) |
| `token_sink` | `TokenSink \| None` | Token streaming sink (discards when None) |
| `hooks` | `LifecycleHook \| None` | Lifecycle hooks (proceeds when None) |

## Custom observer

Subclass `NullObserver` and override only the events you need:

```python
from little_harness.application.ports import AgentObserver
from little_harness.infrastructure.observability.null_observer import NullObserver


class MyObserver(NullObserver):
    def on_run_started(self, run_id, prompt, max_iterations):
        print(f"Run {run_id} started with prompt: {prompt.value}")

    def on_run_finished(self, run_id, result):
        print(f"Run {run_id} finished in {result.elapsed.value:.2f}s")
```

## Custom lifecycle hook

Subclass `NullHook` and override only the events you need:

```python
from little_harness.domain.hook_decision import Proceed
from little_harness.infrastructure.hooks.null_hook import NullHook


class TimingHook(NullHook):
    def __init__(self):
        self.start = None

    def on_session_start(self, run_id, prompt):
        self.start = time.monotonic()
        return Proceed()

    def on_session_end(self, run_id, result):
        if self.start is not None:
            elapsed = time.monotonic() - self.start
            print(f"Session took {elapsed:.2f}s")
```

### Hook decision types

| Decision | Effect |
|----------|--------|
| `Proceed()` | Continue unchanged |
| `InjectContext(content)` | Append message content, then continue |
| `Block(reason)` | Abort/skip with the given reason |

## Interactive console programmatically

```python
from little_harness.composition import (
    build_application,
    build_command_registry,
    build_observer,
    AppConfig,
)
from little_harness.presentation.cli.interactive_console import InteractiveConsole

config = AppConfig(provider="litellm", provider_options={"model": "gemini/gemini-2.5-flash"})
observer = build_observer(config)

with build_application(config, observer) as app:
    registry = build_command_registry()
    InteractiveConsole(app, registry=registry).start()
```

See the {doc}`../plugins/creating-provider` guide for implementing your own
provider plugin, or {doc}`../plugins/creating-tool` for a custom tool.
