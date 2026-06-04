"""Port for intercepting the agent loop — the seam for control-flow hooks.

Unlike `AgentObserver`, which only observes, a `LifecycleHook` returns a
`HookDecision` that can change what happens next. The runtime applies the same
rule at every point:

- `Proceed`        — do the action unchanged.
- `InjectContext`  — append the content as a message, then do the action.
- `Block`          — do not do the action; append the reason and take the
  alternative path (abort the run, skip the tool, or keep looping).

Authors subclass `NullHook` and override only the events they care about, so
this six-method contract never forces an all-or-nothing implementation.
"""

from __future__ import annotations

from typing import Protocol

from local_llm.domain.decision import ToolCall
from local_llm.domain.hook_decision import HookDecision
from local_llm.domain.result import AgentResult
from local_llm.domain.tool_result import ToolRunResult
from local_llm.domain.values.numeric_values import Iteration
from local_llm.domain.values.text_values import MessageContent, Prompt, RunId


class LifecycleHook(Protocol):
    # Arguments are passed positionally, so a hook (or the Null Object) need not
    # echo every name; subclass `NullHook` and override only what you need.
    def on_session_start(self, run_id: RunId, prompt: Prompt, /) -> HookDecision:
        """Decide before the run begins. Block aborts the run with the reason."""
        ...

    def on_user_prompt_submit(self, run_id: RunId, prompt: Prompt, /) -> HookDecision:
        """Decide as the prompt is incorporated. Block aborts with the reason."""
        ...

    def on_pre_tool_use(
        self, run_id: RunId, iteration: Iteration, call: ToolCall, /
    ) -> HookDecision:
        """Decide before a tool runs. Block skips it with the reason as output."""
        ...

    def on_post_tool_use(
        self,
        run_id: RunId,
        iteration: Iteration,
        call: ToolCall,
        result: ToolRunResult,
        /,
    ) -> HookDecision:
        """Decide after a tool ran. Inject/Block append feedback for the model."""
        ...

    def on_stop(
        self, run_id: RunId, iteration: Iteration, answer: MessageContent, /
    ) -> HookDecision:
        """Decide whether to stop. Block keeps looping with the reason."""
        ...

    def on_session_end(self, run_id: RunId, result: AgentResult, /) -> None:
        """Release any resources the hook holds. Cannot change the result."""
        ...
