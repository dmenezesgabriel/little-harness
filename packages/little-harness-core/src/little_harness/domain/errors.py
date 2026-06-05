"""Domain errors raised when agent or tool protocol expectations are violated."""

from __future__ import annotations


class AgentProtocolError(ValueError):
    """Raised when model output cannot be parsed into a valid agent decision."""


class ToolRegistrationError(ValueError):
    """Raised when a tool cannot be registered (empty or duplicate name)."""


class UnknownProviderError(ValueError):
    """Raised when no installed plugin registers the requested provider name."""


class UnknownToolError(ValueError):
    """Raised when `--tools` names a tool no installed plugin registers."""


class UnknownPolicyError(ValueError):
    """Raised when `--policy` names a policy no installed plugin registers."""


class UnknownObserverError(ValueError):
    """Raised when `--observer` names an observer no installed plugin registers."""
