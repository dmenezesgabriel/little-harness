"""No-op `LifecycleHook`: the default when no hooks are configured.

Also the extension point: subclass and override only the events you care about,
the way `NullObserver` keeps `AgentObserver` implementations selective.
"""

from __future__ import annotations

from little_harness.domain.hook_decision import HookDecision, Proceed


class NullHook:
    """Proceeds at every decision point and ends the session silently.

    The decision methods accept and ignore their arguments — the Null Object
    pattern — which is why the signatures are deliberately variadic.
    """

    def on_session_start(self, *_args: object, **_kwargs: object) -> HookDecision:
        return Proceed()

    def on_user_prompt_submit(self, *_args: object, **_kwargs: object) -> HookDecision:
        return Proceed()

    def on_pre_tool_use(self, *_args: object, **_kwargs: object) -> HookDecision:
        return Proceed()

    def on_post_tool_use(self, *_args: object, **_kwargs: object) -> HookDecision:
        return Proceed()

    def on_stop(self, *_args: object, **_kwargs: object) -> HookDecision:
        return Proceed()

    def on_session_end(self, *_args: object, **_kwargs: object) -> None: ...
