# Lifecycle Hooks

A `LifecycleHook` intercepts the agent loop at six decision points and can
change what happens next — unlike `AgentObserver`, which only observes.

## Hook points

| Point | Signature | Block effect |
|-------|-----------|-------------|
| `on_session_start(run_id, prompt)` | `-> HookDecision` | Aborts the run with the reason |
| `on_user_prompt_submit(run_id, prompt)` | `-> HookDecision` | Aborts the run with the reason |
| `on_pre_tool_use(run_id, iteration, call)` | `-> HookDecision` | Skips the tool, reason becomes observation |
| `on_post_tool_use(run_id, iteration, call, result)` | `-> HookDecision` | Injects/block feedback for the model |
| `on_stop(run_id, iteration, answer)` | `-> HookDecision` | Block keeps looping |
| `on_session_end(run_id, result)` | `-> None` | Release-only; cannot change result |

## Decision types

| Decision | Meaning |
|----------|---------|
| `Proceed()` | Continue unchanged |
| `InjectContext(MessageContent)` | Append content as a message, then continue |
| `Block(MessageContent)` | Do not proceed; alternative path with reason |

### Fold semantics (multiple hooks via HookChain)

When multiple hooks are composed in a `HookChain`, decisions fold:

1. First `Block` short-circuits all later hooks
2. All `InjectContext` values are concatenated
3. All hooks `Proceed` → the chain proceeds

## NullHook

Subclass `NullHook` and override only the events you need:

```python
from little_harness.infrastructure.hooks.null_hook import NullHook
from little_harness.domain.hook_decision import Proceed, Block


class AllowlistHook(NullHook):
    def __init__(self, allowed_tools: frozenset[str]) -> None:
        self._allowed = allowed_tools

    def on_pre_tool_use(self, run_id, iteration, call):
        if call.tool_name.value in self._allowed:
            return Proceed()
        return Block(MessageContent(f"Tool {call.tool_name.value!r} not allowed"))
```

## ApprovalHook

The built-in `ApprovalHook` gates tools declared with `requires_approval=True`.
It asks the configured `PermissionRequester` and returns `Block` on rejection.

```python
ApprovalHook(
    requester=InteractivePermissionRequester(),
    names_requiring_approval=frozenset({"bash", "write_file", "edit_file"}),
)
```
