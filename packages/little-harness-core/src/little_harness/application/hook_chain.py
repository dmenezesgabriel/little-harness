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
        self._hooks = tuple(hooks)

    def on_session_start(self, run_id: RunId, prompt: Prompt) -> HookDecision:
        return self._fold(lambda hook: hook.on_session_start(run_id, prompt))

    def on_user_prompt_submit(self, run_id: RunId, prompt: Prompt) -> HookDecision:
        return self._fold(lambda hook: hook.on_user_prompt_submit(run_id, prompt))

    def on_pre_tool_use(
        self, run_id: RunId, iteration: Iteration, call: ToolCall
    ) -> HookDecision:
        return self._fold(lambda hook: hook.on_pre_tool_use(run_id, iteration, call))

    def on_post_tool_use(
        self,
        run_id: RunId,
        iteration: Iteration,
        call: ToolCall,
        result: ToolRunResult,
    ) -> HookDecision:
        return self._fold(
            lambda hook: hook.on_post_tool_use(run_id, iteration, call, result)
        )

    def on_stop(
        self, run_id: RunId, iteration: Iteration, answer: MessageContent
    ) -> HookDecision:
        return self._fold(lambda hook: hook.on_stop(run_id, iteration, answer))

    def on_session_end(self, run_id: RunId, result: AgentResult) -> None:
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
