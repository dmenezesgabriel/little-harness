# little-harness-core

The core of [little-harness](https://github.com/dmenezesgabriel/little-harness):
a provider-agnostic reason-act agent loop (strict-JSON protocol, tool calling,
streaming, lifecycle hooks) and the `little-harness` CLI.

It imports no model vendor SDK. Chat-model providers and tools are separate
distributions, discovered at runtime via packaging entry points, so the core
stays free of any provider dependency until you select one that is installed.

```
uv pip install "little-harness[llama-cpp]"   # core + a provider, via the umbrella
```

Core follows Clean Architecture — `domain`, `application` (the loop plus the
`ChatModel`/`AgentTool`/`AgentPolicy`/`AgentObserver`/`TokenSink`/`LifecycleHook`
ports), `infrastructure` (strict-JSON policy, structured logging), and
`presentation` (CLI) — with dependencies pointing only inward, enforced by
`import-linter`. See the [repository README](https://github.com/dmenezesgabriel/little-harness)
for the full architecture and how to add a provider or tool.
