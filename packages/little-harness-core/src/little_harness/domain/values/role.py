"""Chat role value object with the three valid roles as singletons."""

from __future__ import annotations

from dataclasses import dataclass

VALID_ROLE_NAMES = ("system", "user", "assistant")


@dataclass(frozen=True)
class Role:
    """A chat message role. Hashable, so it can key per-role strategies.

    Use the module singletons (SYSTEM, USER, ASSISTANT) rather than constructing
    ad hoc instances.

    Example:
        message_role = SYSTEM

    """

    name: str

    def __post_init__(self) -> None:
        """Validate that role name is one of the valid role names."""
        if self.name in VALID_ROLE_NAMES:
            return

        raise ValueError(
            f"Invalid role: {self.name}. Expected one of {VALID_ROLE_NAMES}."
        )


SYSTEM = Role("system")
USER = Role("user")
ASSISTANT = Role("assistant")
