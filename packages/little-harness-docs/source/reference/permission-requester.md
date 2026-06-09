# Permission Requester

The `PermissionRequester` port gates sensitive tool calls behind human approval.
When a tool declares `requires_approval=True`, the `ApprovalHook` asks the
configured requester before executing it.

## Port

```python
class PermissionRequester(Protocol):
    def request_approval(self, call: ToolCall, /) -> bool: ...
```

Return `True` to allow the call, `False` to reject it.

## Built-in implementations

| Implementation | When used |
|----------------|-----------|
| `AutoApprovePermissionRequester` | `--yes` flag, piped stdin, or non-interactive runs |
| `InteractivePermissionRequester` | Default interactive REPL (`--ui default`) |
| `RichPermissionRequester` | `--ui rich` via the `little_harness.ui_permission_requesters` entry point |

The auto-approve requester grants every request. The interactive requester
prompts `[y/N]` and defaults to deny on EOF.

## Approval resolution order

```
config.approve_all?  ──yes──> AutoApprove
no
stdin is a tty?      ──no──> AutoApprove
yes
ui != "default"?     ──yes──> discover_permission_requester(ui)  (Rich, etc.)
no
                     InteractivePermissionRequester
```

## Token Sink

The `TokenSink` port streams generated tokens to the user interface as they are
produced, separate from the `AgentObserver` (which is for logging/metrics).

```python
class TokenSink(Protocol):
    def emit(self, chunk: MessageContent) -> None: ...
```

### Built-in implementations

| Implementation | When used |
|----------------|-----------|
| `NullTokenSink` | Streaming disabled (default) — discards every chunk |
| `StdoutTokenSink` | `--stream` flag — writes each chunk to stdout with flush |
