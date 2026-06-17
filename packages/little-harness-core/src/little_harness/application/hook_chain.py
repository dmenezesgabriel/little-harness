"""Composite `LifecycleHook` that runs several hooks and folds their decisions.

The fold gives every point deterministic combined semantics: the first `Block`
short-circuits the rest, otherwise every `InjectContext` is concatenated. This
is the one place that knows how multiple hooks combine, analogous to how
`ToolRegistry` is the one place that owns the name->tool mapping.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from little_harness.application.ports.lifecycle_hook import LifecycleHook
from little_harness.domain.decision import ToolCall
from little_harness.domain.hook_decision import (
    Block,
    HookDecision,
    InjectContext,
    Proceed,
)
from little_harness.domain.message_history import MessageHistory
from little_harness.domain.result import AgentResult
from little_harness.domain.tool_result import ToolRunResult
from little_harness.domain.values.numeric_values import Iteration
from little_harness.domain.values.text_values import MessageContent, Prompt, RunId

HookCall = Callable[[LifecycleHook], HookDecision]


class HookChain:
    """Runs each hook in order and folds the result; itself a `LifecycleHook`.

    Example:
        chain = HookChain([audit_hook, allowlist_hook])
        decision = chain.on_pre_tool_use(run_id, iteration, call)

    """

    def __init__(self, hooks: Sequence[LifecycleHook]) -> None:
        """See class docstring for argument descriptions."""
        self._hooks = tuple(hooks)

    def on_session_start(self, run_id: RunId, prompt: Prompt) -> HookDecision:
        """Fold `on_session_start` across all hooks."""

        def run_hook(hook: LifecycleHook) -> HookDecision:
            return hook.on_session_start(run_id, prompt)

        return self._fold(run_hook)

    def on_user_prompt_submit(self, run_id: RunId, prompt: Prompt) -> HookDecision:
        """Fold `on_user_prompt_submit` across all hooks."""

        def run_hook(hook: LifecycleHook) -> HookDecision:
            return hook.on_user_prompt_submit(run_id, prompt)

        return self._fold(run_hook)

    def on_turn_start(
        self, run_id: RunId, iteration: Iteration, prompt: Prompt
    ) -> HookDecision:
        """Fold `on_turn_start` across all hooks."""

        def run_hook(hook: LifecycleHook) -> HookDecision:
            return hook.on_turn_start(run_id, iteration, prompt)

        return self._fold(run_hook)

    def on_turn_end(
        self, run_id: RunId, iteration: Iteration, output: MessageContent
    ) -> HookDecision:
        """Fold `on_turn_end` across all hooks."""

        def run_hook(hook: LifecycleHook) -> HookDecision:
            return hook.on_turn_end(run_id, iteration, output)

        return self._fold(run_hook)

    def on_model_request(
        self, run_id: RunId, iteration: Iteration
    ) -> HookDecision:
        """Fold `on_model_request` across all hooks."""

        def run_hook(hook: LifecycleHook) -> HookDecision:
            return hook.on_model_request(run_id, iteration)

        return self._fold(run_hook)

    def on_model_response(
        self, run_id: RunId, iteration: Iteration, output: MessageContent
    ) -> HookDecision:
        """Fold `on_model_response` across all hooks."""

        def run_hook(hook: LifecycleHook) -> HookDecision:
            return hook.on_model_response(run_id, iteration, output)

        return self._fold(run_hook)

    def on_context_build(
        self, run_id: RunId, iteration: Iteration, messages: MessageHistory
    ) -> HookDecision:
        """Fold `on_context_build` across all hooks."""

        def run_hook(hook: LifecycleHook) -> HookDecision:
            return hook.on_context_build(run_id, iteration, messages)

        return self._fold(run_hook)

    def on_pre_tool_use(
        self, run_id: RunId, iteration: Iteration, call: ToolCall
    ) -> HookDecision:
        """Fold `on_pre_tool_use` across all hooks."""

        def run_hook(hook: LifecycleHook) -> HookDecision:
            return hook.on_pre_tool_use(run_id, iteration, call)

        return self._fold(run_hook)

    def on_post_tool_use(
        self,
        run_id: RunId,
        iteration: Iteration,
        call: ToolCall,
        result: ToolRunResult,
    ) -> HookDecision:
        """Fold `on_post_tool_use` across all hooks."""

        def run_hook(hook: LifecycleHook) -> HookDecision:
            return hook.on_post_tool_use(run_id, iteration, call, result)

        return self._fold(run_hook)

    def on_stop(
        self, run_id: RunId, iteration: Iteration, answer: MessageContent
    ) -> HookDecision:
        """Fold `on_stop` across all hooks."""

        def run_hook(hook: LifecycleHook) -> HookDecision:
            return hook.on_stop(run_id, iteration, answer)

        return self._fold(run_hook)

    def on_session_end(self, run_id: RunId, result: AgentResult) -> None:
        """Run `on_session_end` on every hook in order."""
        for hook in self._hooks:
            hook.on_session_end(run_id, result)

    def _fold(self, call: HookCall) -> HookDecision:
        folder = _DecisionFolder()

        for hook in self._hooks:
            if not folder.absorb(call(hook)):
                break

        return folder.result()


class _DecisionFolder:
    """Accumulates decisions: keeps injections, stops at the first block.

    Implements `HookDecisionVisitor[bool]`; `absorb` returns False to signal the
    chain to stop calling later hooks.
    """

    def __init__(self) -> None:
        self._injected: list[str] = []
        self._blocked: Block | None = None

    def absorb(self, decision: HookDecision) -> bool:
        return decision.accept(self)

    def visit_proceed(self, _decision: Proceed) -> bool:
        return True

    def visit_inject_context(self, decision: InjectContext) -> bool:
        self._injected.append(decision.content.value)
        return True

    def visit_block(self, decision: Block) -> bool:
        self._blocked = decision
        return False

    def result(self) -> HookDecision:
        if self._blocked is not None:
            return self._blocked

        if not self._injected:
            return Proceed()

        return InjectContext(MessageContent("\n".join(self._injected)))
