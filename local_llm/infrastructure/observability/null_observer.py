"""No-op `AgentObserver`: the default when observability is not configured."""

from __future__ import annotations


class NullObserver:
    """Discards every event so the runtime can always hold a real observer.

    Each method accepts and ignores any arguments — the Null Object pattern —
    which is why the signatures are deliberately variadic.
    """

    def on_run_started(self, *_args: object, **_kwargs: object) -> None: ...

    def on_model_completed(self, *_args: object, **_kwargs: object) -> None: ...

    def on_decision_parsed(self, *_args: object, **_kwargs: object) -> None: ...

    def on_tool_invoked(self, *_args: object, **_kwargs: object) -> None: ...

    def on_repair(self, *_args: object, **_kwargs: object) -> None: ...

    def on_run_finished(self, *_args: object, **_kwargs: object) -> None: ...
