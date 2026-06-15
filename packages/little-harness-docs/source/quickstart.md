# Quickstart

## CLI

One-shot mode (requires `-p`):

```bash
little-harness --model models/LFM2.5-8B-A1B-Q4_K_M.gguf \
  -p "What is 144 divided by 12? Then tell me if the result is even or odd."
```

Interactive REPL mode (omit `-p`):

```bash
little-harness --model models/LFM2.5-8B-A1B-Q4_K_M.gguf
```

With a remote provider:

```bash
export GEMINI_API_KEY="…"

little-harness --provider litellm --model gemini/gemini-2.5-flash \
  -p "What is 144 divided by 12?"
```

Enable streaming and structured logging:

```bash
little-harness --provider litellm --model gemini/gemini-2.5-flash \
  -p "Hello!" --stream --log
```

## Programmatic

```python
from little_harness.composition import run_cli

result = run_cli([
    "--provider", "litellm",
    "--model", "gemini/gemini-2.5-flash",
    "--prompt", "What is 144 divided by 12?",
])
print(result)
```

Or wire the runtime yourself:

```python
from little_harness.application import AgentRuntime, AgentDependencies, AgentRuntimeConfig, ToolRegistry
from little_harness.domain.values.numeric_values import Temperature, MaxTokens, MaxIterations
from little_harness.domain.values.text_values import Prompt
from little_harness.domain.values.truncation import TruncationConfig
from little_harness.infrastructure.hooks.null_hook import NullHook
from little_harness.infrastructure.observability.null_observer import NullObserver
from little_harness.infrastructure.truncation.head_truncator import HeadTruncator
from little_harness.plugin_discovery import (
    load_chat_model_builder,
    discover_policy,
    discover_tools,
)
from little_harness.presentation.cli.token_sinks import NullTokenSink

model = load_chat_model_builder("litellm")({"model": "gemini/gemini-2.5-flash"})
policy = discover_policy("json")
tools = ToolRegistry(discover_tools())

deps = AgentDependencies(
    chat_model=model,
    tool_registry=tools,
    policy=policy,
    observer=NullObserver(),
    token_sink=NullTokenSink(),
    hooks=NullHook(),
    truncator=HeadTruncator(),
    truncation_config=TruncationConfig(),
)
config = AgentRuntimeConfig(
    max_iterations=MaxIterations(5),
    temperature=Temperature(0.0),
    max_tokens=MaxTokens(512),
)
runtime = AgentRuntime(deps, config)

result = runtime.run(Prompt("What is 144 divided by 12?"))
print(result.answer)
```

See {doc}`usage/cli` for the full CLI reference and {doc}`usage/programmatic`
for deeper API usage.
