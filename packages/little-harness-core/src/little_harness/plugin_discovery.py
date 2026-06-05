"""Runtime discovery of provider, tool, policy, and observer plugins via entry points.

This is the single module allowed to import plugin code dynamically: a plugin
distribution registers an entry point, and `entry_point.load()` imports its
adapter (and any vendor SDK) only when selected. Everything else stays
statically typed; the `Any` returned by `.load()` is validated here and nowhere
else, so installing a plugin you do not use never imports its dependencies.

Example:
    builder = load_chat_model_builder("llama_cpp")   # imports the llama adapter
    chat_model = builder({"model_path": "models/m.gguf"})
    policy = discover_policy("json")                 # imports the json policy
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from importlib.metadata import EntryPoint, entry_points
from typing import Any

from little_harness.application.ports.agent_observer import AgentObserver
from little_harness.application.ports.agent_policy import AgentPolicy
from little_harness.application.ports.agent_tool import AgentTool
from little_harness.application.ports.chat_model import ChatModel
from little_harness.domain.errors import (
    UnknownObserverError,
    UnknownPolicyError,
    UnknownProviderError,
    UnknownToolError,
)

PROVIDER_GROUP = "little_harness.chat_model_providers"
TOOL_GROUP = "little_harness.tools"
POLICY_GROUP = "little_harness.agent_policies"
OBSERVER_GROUP = "little_harness.observers"

# A provider plugin exposes this: build a ready ChatModel from its own options.
ChatModelBuilder = Callable[[Mapping[str, str]], ChatModel]


def load_chat_model_builder(name: str) -> ChatModelBuilder:
    # Providers take options, so the deferred builder is returned for the caller
    # to invoke; policy/observer builders take none and are invoked immediately.
    return resolve_builder(PROVIDER_GROUP, name, UnknownProviderError, "provider")


def discover_policy(name: str) -> AgentPolicy:
    """Build the policy plugin registered under `name`.

    Example:
        policy = discover_policy("json")
    """
    return resolve_builder(POLICY_GROUP, name, UnknownPolicyError, "policy")()


def discover_observer(name: str) -> AgentObserver:
    """Build the observer plugin registered under `name`.

    Example:
        observer = discover_observer("logging")
    """
    return resolve_builder(OBSERVER_GROUP, name, UnknownObserverError, "observer")()


def resolve_builder(
    group: str, name: str, error_type: type[ValueError], kind: str
) -> Any:
    """Load the callable a plugin registered under `name`, validated, still `Any`.

    `entry_point.load()` returns `Any`; the callable check lives in a helper so
    the `Any` is not narrowed here and is returned for the caller to type. This
    is the single dynamic-typing boundary, validated here and nowhere else.

    Example:
        builder = resolve_builder(POLICY_GROUP, "json", UnknownPolicyError, "policy")
    """
    matches = entry_points(group=group, name=name)

    if not matches:
        raise error_type(
            f"Unknown {kind}: {name!r}. "
            f"Installed {kind} plugins: {installed_names(group)}."
        )

    builder = next(iter(matches)).load()
    require_callable_builder(builder, name, error_type, kind)
    return builder


def require_callable_builder(
    builder: object, name: str, error_type: type[ValueError], kind: str
) -> None:
    if callable(builder):
        return

    raise error_type(
        f"{kind.capitalize()} {name!r} did not load a callable builder: {builder!r}."
    )


def require_sole_installed(
    group: str, error_type: type[ValueError], kind: str, flag: str
) -> str:
    """Name the only installed plugin in `group`, or fail with the installed list.

    Zero or several is ambiguous: the caller must pass `flag` to disambiguate.
    """
    installed = installed_names(group)

    if len(installed) != 1:
        raise error_type(
            f"No {kind} selected and {len(installed)} installed: {installed}. "
            f"Pass {flag}, or install exactly one {kind} plugin."
        )

    return installed[0]


def installed_names(group: str) -> list[str]:
    return sorted(point.name for point in entry_points(group=group))


def installed_providers() -> list[str]:
    return installed_names(PROVIDER_GROUP)


def default_provider_name() -> str:
    """Name the provider to use when `--provider` is omitted.

    Core ships no provider, so the default is whichever single provider is
    installed. Zero or several is ambiguous and fails with the installed list.

    Example:
        builder = load_chat_model_builder(default_provider_name())
    """
    return require_sole_installed(
        PROVIDER_GROUP, UnknownProviderError, "provider", "--provider"
    )


def default_policy_name() -> str:
    """Name the policy to use when `--policy` is omitted.

    Core ships no policy, so the default is whichever single policy is installed.
    Zero or several is ambiguous and fails with the installed list.

    Example:
        policy = discover_policy(default_policy_name())
    """
    return require_sole_installed(
        POLICY_GROUP, UnknownPolicyError, "policy", "--policy"
    )


def installed_tools() -> list[str]:
    return installed_names(TOOL_GROUP)


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
