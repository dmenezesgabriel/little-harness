# Programmatic usage

## The composition root

The simplest way to use `little-harness` from code is through
`run_cli` — it parses a `sys.argv`-style list and returns rendered text:

```python
from little_harness.composition import run_cli

text = run_cli(["--prompt", "Hello!"])
print(text)
```

## The runtime directly

For full control, wire the `AgentRuntime` yourself:

```python
from little_harness.application import AgentRuntime, AgentDependencies, ToolRegistry
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

runtime = AgentRuntime(deps)
result = runtime.run("What is 144 divided by 12?")

print(f"Answer: {result.answer}")
print(f"Elapsed: {result.elapsed:.2f}s")
print(f"Steps: {len(result.steps)}")
```

## Custom observer

```python
from little_harness.application.ports import AgentObserver
from little_harness.application import AgentDependencies, AgentRuntime

class MyObserver(AgentObserver):
    def on_run_started(self, run_id, prompt, max_iterations):
        print(f"Run {run_id} started")

    def on_run_finished(self, run_id, result):
        print(f"Run {run_id} finished in {result.elapsed:.2f}s")

    # ...other lifecycle hooks (all are no-op by default)

deps = AgentDependencies(..., observer=MyObserver())
```

## Custom lifecycle hook

```python
from little_harness.domain.hook_decision import HookDecision, Proceed
from little_harness.application.ports import LifecycleHook

class TimingHook(LifecycleHook):
    def __init__(self):
        self.start = None

    def on_session_start(self):
        self.start = time.monotonic()
        return Proceed()

    def on_session_end(self):
        elapsed = time.monotonic() - self.start
        print(f"Session took {elapsed:.2f}s")
```

See the {doc}`../plugins/creating-provider` guide for implementing your own
provider plugin, or {doc}`../plugins/creating-tool` for a custom tool.
