# Architecture overview

## Package dependency graph

```
little-harness (umbrella, no code)
  └── little-harness-core       # provider-agnostic, zero deps
        ├── little-harness-llama-cpp   (chat_model_providers)
        ├── little-harness-litellm     (chat_model_providers)
        ├── little-harness-calculator  (tools)
        ├── little-harness-file-tools  (tools)
        ├── little-harness-ripgrep     (tools)
        ├── little-harness-ast         (tools)
        ├── little-harness-json-policy (agent_policies)
        └── little-harness-logging     (observers)
```

All plugin packages depend on `little-harness-core` only. No plugin depends
on another plugin. Core depends on nothing.

## Layers (Clean Architecture)

`little-harness-core` follows Clean Architecture with four concentric layers:

```
┌──────────────────────────────┐
│        presentation/         │  CLI: args, rendering
│   (argument_parser, __main__)│
├──────────────────────────────┤
│      infrastructure/         │  Core defaults: NullObserver, hooks
│   (observability, hooks)     │
├──────────────────────────────┤
│       application/           │  AgentRuntime loop, ports, registries
│   (agent_runtime, ports)     │
├──────────────────────────────┤
│         domain/              │  Pure entities, value objects, errors
│   (decision, message, values)│
└──────────────────────────────┘
```

Dependencies point **inward** only — `presentation` and `infrastructure` may
depend on `application` and `domain`; `application` may depend on `domain`;
`domain` depends on nothing. This is mechanically enforced by `import-linter`
for every package.

## Agent loop

```
User -> AgentRuntime : run(prompt)
AgentRuntime -> AgentPolicy : system_prompt(tools)
AgentRuntime -> ChatModel : complete_streaming(request)
ChatModel -> AgentRuntime : tokens
AgentRuntime -> AgentPolicy : parse_model_output(text)

alt ToolCall
  AgentPolicy -> AgentRuntime : ToolCall
  AgentRuntime -> AgentTool : run(request)
  AgentTool -> AgentRuntime : ToolRunResult
  AgentRuntime -> AgentPolicy : build_tool_observation(result)
  AgentRuntime -> ChatModel : complete_streaming(next request)
else FinalAnswer
  AgentPolicy -> AgentRuntime : FinalAnswer
  AgentRuntime -> User : AgentResult(answer)
end
```

## Plugin discovery

All dynamic imports happen in `little_harness/plugin_discovery.py` via
`importlib.metadata.entry_points()`. The entry-point groups are:

| Group | Purpose | Builder signature |
|-------|---------|-------------------|
| `little_harness.chat_model_providers` | Chat model adapters | `build(options: dict) -> ChatModel` |
| `little_harness.tools` | Agent tools | `build() -> AgentTool` |
| `little_harness.agent_policies` | Agent policies | `build() -> AgentPolicy` |
| `little_harness.observers` | Observers | `build() -> AgentObserver` |

A plugin's vendor SDK is imported **only when the plugin is selected**
(lazy loading).
