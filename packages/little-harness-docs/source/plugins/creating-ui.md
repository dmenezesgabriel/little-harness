# Creating a UI plugin

A UI plugin handles the interactive REPL execution by implementing the `InteractiveRunner` protocol.

## Implement `InteractiveRunner`

An interactive UI plugin must implement the `InteractiveRunner` protocol, which defines a single `start` method.

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

The entry point must point to a builder function with the following signature:

```python
def build(
    app: Application,
    registry: CommandRegistry,
) -> InteractiveRunner:
    """Build and return an InteractiveRunner instance.

    Usage example:
        runner = build(app, registry)
    """
    return MyCustomConsole(app, registry)
```

## Register the Entry Point

Register your UI builder under the `little_harness.uis` entry-point group in `pyproject.toml`:

```toml
[project.entry-points."little_harness.uis"]
custom = "little_harness_custom_ui.provider:build"
```
