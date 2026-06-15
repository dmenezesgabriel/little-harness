# Plugin system

## Entry-point registration

Each plugin distribution registers builder callables in its `pyproject.toml`:

```toml
[project.entry-points."little_harness.chat_model_providers"]
llama_cpp = "little_harness_llama_cpp.provider:build"
litellm   = "little_harness_litellm.provider:build"
```

```toml
[project.entry-points."little_harness.tools"]
calculator   = "little_harness_calculator.provider:build"
read_file    = "little_harness_file_tools.provider:build_read_file"
write_file   = "little_harness_file_tools.provider:build_write_file"
edit_file    = "little_harness_file_tools.provider:build_edit_file"
bash         = "little_harness_file_tools.provider:build_bash"
ripgrep      = "little_harness_ripgrep.provider:build"
ast_grep     = "little_harness_ast.provider:build_ast_grep"
ast_edit     = "little_harness_ast.provider:build_ast_edit"
```

```toml
[project.entry-points."little_harness.agent_policies"]
json = "little_harness_json_policy.provider:build"
```

```toml
[project.entry-points."little_harness.observers"]
logging = "little_harness_logging.provider:build"
```

```toml
[project.entry-points."little_harness.uis"]
rich = "little_harness_rich.provider:build"
```

```toml
[project.entry-points."little_harness.ui_permission_requesters"]
rich = "little_harness_rich.provider:build_permission_requester"
```

```toml
[project.entry-points."little_harness.repl_commands"]
# No built-in plugin commands; packages may register their own.
```

```toml
[project.entry-points."little_harness.session_plugins"]
jsonl = "little_harness_session_jsonl.plugin:build_plugin"
```

## Discovery

The discovery module (`little_harness/plugin_discovery.py`) is the single
dynamic-import boundary. Every other module imports it, not individual plugins.

```python
from little_harness.plugin_discovery import (
    load_chat_model_builder,
    discover_tools,
    discover_policy,
    discover_observer,
    discover_ui,
    discover_permission_requester,
    discover_repl_commands,
    installed_providers,
    installed_tools,
)
```

Key design rules:
- `entry_point.load()` returns `Any` — the only untyped seam; validated
  immediately by `require_callable_builder()`.
- With exactly one plugin installed in a group, the `--provider`/`--policy`
  flags are optional (auto-detected as the sole member).
- With zero or several, omitting the flag raises a clear error listing
  installed plugins.

## Ports (interfaces)

All ports are `typing.Protocol` classes:

| Port | Required method | Used by |
|------|-----------------|---------|
| `ChatModel` | `complete_streaming(request)` + `supports_thinking()` | AgentRuntime |
| `AgentTool` | `spec -> ToolSpec` + `run(request) -> ToolRunResult` | AgentRuntime |
| `AgentPolicy` | `system_prompt(tools)`, `parse_model_output()`, `build_repair_message()` | AgentRuntime |
| `AgentObserver` | `on_run_started()`, `on_model_completed()`, … | AgentRuntime |
| `TokenSink` | `emit(chunk)` | AgentRuntime |
| `ToolTruncator` | `truncate(content, config) -> TruncationResult` | AgentRuntime |
| `LifecycleHook` | `on_session_start()`, `on_pre_tool_use()`, … | Application |
| `PermissionRequester` | `request_approval(call) -> bool` | ApprovalHook |
| `SessionPlugin` | `observer()`, `repository()` | Application / CLI |
| `SessionRepository` | `load(session_id) -> MessageHistory` | InteractiveConsole |
| `InteractiveRunner` | `start() -> str` | CLI / Composition |

See {doc}`../plugins/creating-provider` for implementing each port.
