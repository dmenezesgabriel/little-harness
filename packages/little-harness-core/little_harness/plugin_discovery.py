"""Runtime discovery of provider and tool plugins via packaging entry points.

This is the single module allowed to import plugin code dynamically: a plugin
distribution registers an entry point, and `entry_point.load()` imports its
adapter (and any vendor SDK) only when selected. Everything else stays
statically typed; the `Any` returned by `.load()` is validated here and nowhere
else, so installing a provider you do not use never imports its dependencies.

Example:
    builder = load_chat_model_builder("llama_cpp")   # imports the llama adapter
    chat_model = builder({"model_path": "models/m.gguf"})
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from importlib.metadata import entry_points

from little_harness.application.ports.agent_tool import AgentTool
from little_harness.application.ports.chat_model import ChatModel
from little_harness.domain.errors import UnknownProviderError

PROVIDER_GROUP = "little_harness.chat_model_providers"
TOOL_GROUP = "little_harness.tools"

# A provider plugin exposes this: build a ready ChatModel from its own options.
ChatModelBuilder = Callable[[Mapping[str, str]], ChatModel]


def load_chat_model_builder(name: str) -> ChatModelBuilder:
    matches = entry_points(group=PROVIDER_GROUP, name=name)

    if not matches:
        raise UnknownProviderError(
            f"Unknown provider: {name!r}. Installed providers: {installed_providers()}."
        )

    builder = next(iter(matches)).load()
    require_callable_builder(builder, name)
    # `entry_point.load()` returns `Any`; the callable check lives in a helper so the
    # `Any` is not narrowed here and `builder` is returned as the typed builder. This
    # is the single dynamic-typing boundary, validated here and nowhere else.
    return builder


def require_callable_builder(builder: object, name: str) -> None:
    if not callable(builder):
        raise UnknownProviderError(
            f"Provider {name!r} did not load a callable builder: {builder!r}."
        )


def installed_providers() -> list[str]:
    return sorted(point.name for point in entry_points(group=PROVIDER_GROUP))


def discover_tools() -> Sequence[AgentTool]:
    return [point.load()() for point in entry_points(group=TOOL_GROUP)]
