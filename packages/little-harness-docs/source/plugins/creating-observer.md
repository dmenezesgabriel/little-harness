# Creating an observer plugin

An observer receives lifecycle events from the `AgentRuntime`.

## Implement `AgentObserver`

All methods are optional — subclass `NullObserver` and override only the events
you need.

```python
from little_harness.application.ports.agent_observer import AgentObserver
from little_harness.domain.decision import AgentDecision
from little_harness.domain.result import AgentResult
from little_harness.domain.tool_result import ToolRunResult
from little_harness.domain.values.numeric_values import ElapsedSeconds, Iteration
from little_harness.domain.values.text_values import MessageContent, Prompt, RunId


class MyObserver(AgentObserver):
    def on_run_started(self, run_id: RunId, prompt: Prompt) -> None:
        print(f"Run {run_id} started with prompt: {prompt.value}")

    def on_model_completed(
        self,
        run_id: RunId,
        iteration: Iteration,
        output: MessageContent,
        elapsed: ElapsedSeconds,
    ) -> None:
        print(f"Model output (iteration {iteration}): {output.value}")

    def on_decision_parsed(
        self,
        run_id: RunId,
        iteration: Iteration,
        decision: AgentDecision,
    ) -> None:
        print(f"Decision: {decision.action_name()}")

    def on_tool_invoked(
        self,
        run_id: RunId,
        iteration: Iteration,
        result: ToolRunResult,
        elapsed: ElapsedSeconds,
    ) -> None:
        status = "succeeded" if result.succeeded else "failed"
        print(f"Tool {result.tool_name.value} {status} in {elapsed.value:.2f}s")

    def on_repair(
        self, run_id: RunId, iteration: Iteration, error: Exception
    ) -> None:
        print(f"Repairing: {error}")

    def on_run_finished(self, run_id: RunId, result: AgentResult) -> None:
        print(
            f"Run finished: {result.answer.value} "
            f"({result.elapsed.value:.2f}s, {len(result.steps)} steps)"
        )
```

## Register the entry point

```toml
[project.entry-points."little_harness.observers"]
my_observer = "little_harness_my_observer.provider:build"
```

## The builder

```python
# src/little_harness_my_observer/provider.py
from little_harness.application.ports.agent_observer import AgentObserver
from little_harness_my_observer.my_observer import MyObserver


def build() -> AgentObserver:
    return MyObserver()
```

## NullObserver

Instead of implementing `AgentObserver` directly, subclass `NullObserver`:

```python
from little_harness.infrastructure.observability.null_observer import NullObserver


class MySelectiveObserver(NullObserver):
    def on_run_started(self, run_id: RunId, prompt: Prompt) -> None:
        print(f"Run {run_id} started")
```
