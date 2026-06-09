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
from little_harness.domain.json_object_input import JsonObjectInput
from little_harness.domain.tool_spec import ToolInputSchema
from little_harness.domain.values.text_values import ToolOutput, ToolName


class MyTool(AgentTool):
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=ToolName("my_tool"),
            description="Does something useful.",
            input_schema=ToolInputSchema(
                description="A JSON object with a 'query' string field.",
                json_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                    },
                    "required": ["query"],
                },
            ),
            requires_approval=False,
        )

    def run(self, request: ToolRunRequest) -> ToolRunResult:
        fields = JsonObjectInput.parse(request.tool_input.value)
        query = fields.required_text("query")
        return ToolRunResult(
            tool_name=ToolName("my_tool"),
            output=ToolOutput(f"Processed: {query}"),
            succeeded=True,
        )
```

### For tools with a single string input (no JSON parsing)

```python
class SimpleTool(AgentTool):
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=ToolName("simple"),
            description="Takes a plain string input.",
            input_schema=ToolInputSchema(
                description="A plain text input string.",
                json_schema={
                    "type": "object",
                    "properties": {
                        "input": {"type": "string"},
                    },
                    "required": ["input"],
                },
            ),
            requires_approval=False,
        )

    def run(self, request: ToolRunRequest) -> ToolRunResult:
        return ToolRunResult(
            tool_name=ToolName("simple"),
            output=ToolOutput(f"Received: {request.tool_input.value}"),
            succeeded=True,
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
gates execution through the configured `PermissionRequester`.

```python
ToolSpec(
    name=ToolName("sensitive_tool"),
    description="Can modify system state.",
    input_schema=ToolInputSchema(...),
    requires_approval=True,
)
```

Users bypass the prompt with `--yes`.

## 6. Error handling

Return a failed `ToolRunResult` for recoverable errors (the model will see the
observation and can retry or choose a different tool). Raise exceptions only for
unrecoverable programming errors:

```python
def run(self, request: ToolRunRequest) -> ToolRunResult:
    try:
        return self._do_work(request)
    except ValueError as error:
        return ToolRunResult(
            tool_name=...,
            output=ToolOutput(str(error)),
            succeeded=False,
        )
```

## 7. Testing

```python
from little_harness_my_tool.my_tool import MyTool
from little_harness.domain import ToolSpec, ToolRunRequest, ToolRunResult
from little_harness.domain.values.text_values import ToolInput, ToolName


def test_my_tool_spec() -> None:
    tool = MyTool()
    assert isinstance(tool.spec, ToolSpec)
    assert tool.spec.name.value == "my_tool"


def test_my_tool_run() -> None:
    tool = MyTool()
    request = ToolRunRequest(
        tool_name=ToolName("my_tool"),
        tool_input=ToolInput('{"query": "hello"}'),
    )
    result = tool.run(request)
    assert result.succeeded
    assert "hello" in result.output.value


def test_my_tool_invalid_input() -> None:
    tool = MyTool()
    request = ToolRunRequest(
        tool_name=ToolName("my_tool"),
        tool_input=ToolInput("not json"),
    )
    result = tool.run(request)
    assert not result.succeeded
```
