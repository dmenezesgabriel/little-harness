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

from little_harness.domain.decision import ToolCall
from little_harness.domain.hook_decision import HookDecision
from little_harness.domain.message_history import MessageHistory
from little_harness.domain.result import AgentResult
from little_harness.domain.tool_result import ToolRunResult
from little_harness.domain.values.numeric_values import Iteration
from little_harness.domain.values.text_values import MessageContent, Prompt, RunId


class LifecycleHook(Protocol):
    """Intercept the agent loop at every decision point.

    Arguments are passed positionally, so a hook (or the Null Object) need not
    echo every name; subclass `NullHook` and override only what you need.
    """

    def on_session_start(self, run_id: RunId, prompt: Prompt, /) -> HookDecision:
        """Decide before the run begins. Block aborts the run with the reason."""
        ...

    def on_user_prompt_submit(self, run_id: RunId, prompt: Prompt, /) -> HookDecision:
        """Decide as the prompt is incorporated. Block aborts with the reason."""
        ...

    def on_turn_start(
        self, run_id: RunId, iteration: Iteration, prompt: Prompt, /
    ) -> HookDecision:
        """Decide at the start of each iteration, before the model call.

        Block aborts this iteration with the reason as the answer.
        """
        ...

    def on_turn_end(
        self, run_id: RunId, iteration: Iteration, output: MessageContent, /
    ) -> HookDecision:
        """Decide after the model output, before parsing.

        Block replaces the model output with the reason.
        """
        ...

    def on_model_request(self, run_id: RunId, iteration: Iteration, /) -> HookDecision:
        """Decide right before the model API call.

        Block skips the API call and uses the reason as fake model output.
        """
        ...

    def on_model_response(
        self, run_id: RunId, iteration: Iteration, output: MessageContent, /
    ) -> HookDecision:
        """Decide right after the model API call returns.

        Block replaces the model response with the reason.
        """
        ...

    def on_context_build(
        self, run_id: RunId, iteration: Iteration, messages: MessageHistory, /
    ) -> HookDecision:
        """Decide after model request, before the API call, with full context.

        ``InjectContext`` appends additional context messages before the model
        call. ``Block`` skips the API call same as ``on_model_request``.
        """
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
