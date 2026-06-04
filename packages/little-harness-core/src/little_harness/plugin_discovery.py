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
from importlib.metadata import EntryPoint, entry_points

from little_harness.application.ports.agent_tool import AgentTool
from little_harness.application.ports.chat_model import ChatModel
from little_harness.domain.errors import UnknownProviderError, UnknownToolError

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


def default_provider_name() -> str:
    """Name the provider to use when `--provider` is omitted.

    Core ships no provider, so the default is whichever single provider is
    installed. Zero or several is ambiguous and fails with the installed list.

    Example:
        builder = load_chat_model_builder(default_provider_name())
    """
    installed = installed_providers()

    if len(installed) != 1:
        raise UnknownProviderError(
            f"No provider selected and {len(installed)} installed: {installed}. "
            "Pass --provider, or install exactly one provider plugin."
        )

    return installed[0]


def installed_tools() -> list[str]:
    return sorted(point.name for point in entry_points(group=TOOL_GROUP))


def discover_tools(selection: Sequence[str] | None = None) -> Sequence[AgentTool]:
    """Instantiate registered tool plugins, optionally limited to a selection.

    `selection` is the `--tools` list; None means every installed tool. An
    unknown name fails loudly with the installed list rather than silently
    dropping it.

    Example:
        tools = discover_tools(["read_file", "ripgrep"])
    """
    points = {point.name: point for point in entry_points(group=TOOL_GROUP)}
    names = sorted(points) if selection is None else selection
    return [build_discovered_tool(points, name) for name in names]


def build_discovered_tool(points: Mapping[str, EntryPoint], name: str) -> AgentTool:
    point = points.get(name)

    if point is None:
        raise UnknownToolError(
            f"Unknown tool: {name!r}. Installed tools: {sorted(points)}."
        )

    return point.load()()
