"""No-op `LifecycleHook`: the default when no hooks are configured.

Also the extension point: subclass and override only the events you care about,
the way `NullObserver` keeps `AgentObserver` implementations selective.
Precise signatures (not variadic) let pyright verify subclass overrides.
"""

from __future__ import annotations

from little_harness.domain.decision import ToolCall
from little_harness.domain.hook_decision import HookDecision, Proceed
from little_harness.domain.message_history import MessageHistory
from little_harness.domain.result import AgentResult
from little_harness.domain.tool_result import ToolRunResult
from little_harness.domain.values.numeric_values import Iteration
from little_harness.domain.values.text_values import MessageContent, Prompt, RunId


class NullHook:
    """Proceeds at every decision point and ends the session silently.

    The Null Object of `LifecycleHook`. Subclass and override only the events
    you care about; every other point returns `Proceed` automatically.

    Example:
        class MyHook(NullHook):
            def on_pre_tool_use(
                self, run_id: RunId, iteration: Iteration, call: ToolCall, /
            ) -> HookDecision:
                return Proceed()

    """

    def on_session_start(self, _run_id: RunId, _prompt: Prompt, /) -> HookDecision:
        """Proceed with session start."""
        return Proceed()

    def on_user_prompt_submit(self, _run_id: RunId, _prompt: Prompt, /) -> HookDecision:
        """Proceed with the user prompt submission."""
        return Proceed()

    def on_turn_start(
        self, _run_id: RunId, _iteration: Iteration, _prompt: Prompt, /
    ) -> HookDecision:
        """Proceed with the turn."""
        return Proceed()

    def on_turn_end(
        self, _run_id: RunId, _iteration: Iteration, _output: MessageContent, /
    ) -> HookDecision:
        """Proceed with the turn end."""
        return Proceed()

    def on_model_request(
        self, _run_id: RunId, _iteration: Iteration, /
    ) -> HookDecision:
        """Proceed with the model request."""
        return Proceed()

    def on_model_response(
        self, _run_id: RunId, _iteration: Iteration, _output: MessageContent, /
    ) -> HookDecision:
        """Proceed with the model response."""
        return Proceed()

    def on_context_build(
        self, _run_id: RunId, _iteration: Iteration, _messages: MessageHistory, /
    ) -> HookDecision:
        """Proceed with the context build."""
        return Proceed()

    def on_pre_tool_use(
        self, _run_id: RunId, _iteration: Iteration, _call: ToolCall, /
    ) -> HookDecision:
        """Proceed with the tool call."""
        return Proceed()

    def on_post_tool_use(
        self,
        _run_id: RunId,
        _iteration: Iteration,
        _call: ToolCall,
        _result: ToolRunResult,
        /,
    ) -> HookDecision:
        """Proceed after tool use."""
        return Proceed()

    def on_stop(
        self, _run_id: RunId, _iteration: Iteration, _answer: MessageContent, /
    ) -> HookDecision:
        """Proceed with the stop request."""
        return Proceed()

    def on_session_end(self, _run_id: RunId, _result: AgentResult, /) -> None:
        """Proceed with session end."""
        ...
