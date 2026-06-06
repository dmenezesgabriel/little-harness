# Creating a tool plugin

A tool plugin wraps a capability behind the `AgentTool` port.

## 1. Create the package

```
little-harness-my-tool/
  pyproject.toml
  src/
    little_harness_my_tool/
      __init__.py
      provider.py
      my_tool.py
```

## 2. Implement `AgentTool`

```python
# src/little_harness_my_tool/my_tool.py
from little_harness.application.ports import AgentTool
from little_harness.domain import ToolSpec, ToolRunRequest, ToolRunResult
from little_harness.domain.values import ToolOutput, ToolInput


class MyTool(AgentTool):
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="my_tool",
            description="Does something useful.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
            requires_approval=False,
        )

    def run(self, request: ToolRunRequest) -> ToolRunResult:
        query = request.tool_input.json_object.get_string("query")
        return ToolRunResult(
            output=ToolOutput(f"Processed: {query}"),
            success=True,
        )
```

## 3. Register the entry point

```toml
[project.entry-points."little_harness.tools"]
my_tool = "little_harness_my_tool.provider:build"
```

## 4. The builder

```python
# src/little_harness_my_tool/provider.py
from little_harness.application.ports import AgentTool
from little_harness_my_tool.my_tool import MyTool


def build() -> AgentTool:
    return MyTool()
```

## 5. Approval for sensitive tools

Set `requires_approval=True` in `ToolSpec`. The built-in `ApprovalHook`
will gate execution through the configured `PermissionRequester`.

```python
ToolSpec(
    name="sensitive_tool",
    description="Can modify system state.",
    input_schema=...,
    requires_approval=True,
)
```

Users bypass the prompt with `--yes`.
