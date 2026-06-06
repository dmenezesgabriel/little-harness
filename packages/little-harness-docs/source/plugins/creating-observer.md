# Creating an observer plugin

An observer receives lifecycle events from the `AgentRuntime`.

## Implement `AgentObserver`

All methods are optional — the `NullObserver` provides default no-op
implementations. Override only the events you need.

```python
from little_harness.application.ports import AgentObserver
from little_harness.domain import AgentResult, AgentStep, ChatMessage
from little_harness.domain.values import RunId, Prompt


class MyObserver(AgentObserver):
    def on_run_started(
        self,
        run_id: RunId,
        prompt: Prompt,
        max_iterations: int,
    ) -> None:
        print(f"Run {run_id} started with prompt: {prompt}")

    def on_model_completed(
        self,
        run_id: RunId,
        iteration: int,
        raw_output: str,
    ) -> None:
        print(f"Model output (iteration {iteration}): {raw_output}")

    def on_decision_parsed(
        self,
        run_id: RunId,
        iteration: int,
        decision: AgentDecision,
    ) -> None:
        print(f"Decision: {decision.action_name}")

    def on_tool_invoked(
        self,
        run_id: RunId,
        iteration: int,
        tool_name: str,
        duration: float,
        success: bool,
    ) -> None:
        print(
            f"Tool {tool_name} {'succeeded' if success else 'failed'} "
            f"in {duration:.2f}s"
        )

    def on_repair(
        self,
        run_id: RunId,
        iteration: int,
        raw: str,
        error: str,
    ) -> None:
        print(f"Repairing: {error}")

    def on_run_finished(
        self,
        run_id: RunId,
        result: AgentResult,
    ) -> None:
        print(
            f"Run finished: {result.answer} "
            f"({result.elapsed:.2f}s, {len(result.steps)} steps)"
        )
```

## Register the entry point

```toml
[project.entry-points."little_harness.observers"]
my_observer = "little_harness_my_observer.provider:build"
```
