# little-harness

A small, fully local LLM agent loop with a **plugin architecture**. The core is a
provider-agnostic reason-act loop (strict-JSON protocol, tool calling, streaming,
lifecycle hooks); chat-model providers and tools are separate, independently
installable distributions discovered at runtime via packaging **entry points**.

```{toctree}
:maxdepth: 2
:caption: Getting started

installation
quickstart
usage/cli
usage/programmatic
```

```{toctree}
:maxdepth: 2
:caption: Architecture

architecture/overview
architecture/plugin-system
```

```{toctree}
:maxdepth: 2
:caption: Plugin development

plugins/creating-provider
plugins/creating-tool
plugins/creating-policy
plugins/creating-observer
plugins/creating-ui
plugins/creating-session
```

```{toctree}
:maxdepth: 2
:caption: Reference

reference/lifecycle-hooks
reference/permission-requester
reference/errors
```

```{toctree}
:maxdepth: 2
:caption: Project

contributing
changelog
```

## Key features

- **Zero third-party deps in core** — the base distribution imports nothing
  outside the standard library.
- **Plugin discovery via entry points** — providers, tools, policies, and
  observers are loaded through `importlib.metadata.entry_points()`. Each plugin
  is a separate `pip install`.
- **Clean Architecture** — four concentric layers with dependency inversion
  at the package boundary, mechanically enforced by `import-linter`.
- **Strict-JSON protocol** — the built-in policy drives a typed JSON tool-calling
  schema, parsed and validated on every agent step.
- **Streaming, hooks, structured logging** — `TokenSink`, `LifecycleHook`,
  `AgentObserver` — all pluggable through port interfaces.
- **Safety** — `DangerousCommandGuardrail` blocks destructive shell commands;
  `ApprovalHook` gates sensitive tools via `PermissionRequester`.
