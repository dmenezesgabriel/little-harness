# Creating a UI plugin

A UI plugin handles the interactive REPL display by implementing the `InteractiveRunner` protocol.

## Implement `InteractiveRunner`

```python
from little_harness.presentation.cli.interactive_console import (
    Application,
    InteractiveRunner,
)
from little_harness.presentation.cli.repl_command import CommandRegistry


class MyCustomConsole(InteractiveRunner):
    def __init__(
        self,
        application: Application,
        registry: CommandRegistry,
    ) -> None:
        self.app = application
        self.registry = registry

    def start(self) -> str:
        print("Welcome to My Custom UI!")
        # Implement your interactive loop, reading user inputs and invoking:
        # self.app.run_turn(prompt, message_history)
        return "Finished session"
```

## Implement the Builder

```python
def build(
    app: Application,
    registry: CommandRegistry,
) -> InteractiveRunner:
    return MyCustomConsole(app, registry)
```

## Register the Entry Point

```toml
[project.entry-points."little_harness.uis"]
custom = "little_harness_custom_ui.provider:build"
```

## Permission Requester Integration

If your UI provides styled prompts for tool approvals (e.g. Rich's `Confirm`),
register a `PermissionRequester` under the `little_harness.ui_permission_requesters`
group, keyed to the same name as your UI:

```toml
[project.entry-points."little_harness.ui_permission_requesters"]
custom = "little_harness_custom_ui.provider:build_permission_requester"
```

```python
from little_harness.application.ports.permission_requester import PermissionRequester
from little_harness.domain.decision import ToolCall


class CustomPermissionRequester(PermissionRequester):
    def request_approval(self, call: ToolCall, /) -> bool:
        # Show a styled prompt, return True/False
        ...
```

When `--ui custom` is used, the composition root discovers and uses the matching
permission requester automatically.

## Token Sink Integration

For live streaming display, write incoming tokens to your UI surface. The runtime
emits tokens through a `TokenSink` — in the CLI, `StdoutTokenSink` writes chunks
directly to a stream. Rich UI plugins can instead capture tokens and render them
incrementally as formatted markdown.
