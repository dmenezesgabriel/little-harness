"""Shared validation guards for domain value objects.

Centralizing these keeps range/format rules in one place (DRY) and gives every
value object a consistent exception shape: offending value + expected shape.

Example:
    normalized = require_non_empty_text(" calc ", "ToolName")  # -> "calc"

"""

from __future__ import annotations


def require_non_empty_text(value: str, field_name: str) -> str:
    """Return `value` stripped, or raise if empty."""
    stripped = value.strip()

    if stripped == "":
        raise ValueError(f"{field_name} is empty: {value!r}. Expected non-empty text.")

    return stripped


def require_positive_int(value: int, field_name: str) -> int:
    """Return `value` unchanged, or raise if not > 0."""
    if value > 0:
        return value

    raise ValueError(f"{field_name} is not positive: {value}. Expected an integer > 0.")


def require_non_negative_int(value: int, field_name: str) -> int:
    """Return `value` unchanged, or raise if negative."""
    if value >= 0:
        return value

    raise ValueError(f"{field_name} is negative: {value}. Expected an integer >= 0.")


def require_within(value: float, low: float, high: float, field_name: str) -> float:
    """Return `value` unchanged, or raise if outside `low`..`high`."""
    if low <= value <= high:
        return value

    raise ValueError(f"{field_name} out of range: {value}. Expected {low}..{high}.")


def require_non_negative(value: float, field_name: str) -> float:
    """Return `value` unchanged, or raise if negative."""
    if value >= 0:
        return value

    raise ValueError(f"{field_name} is negative: {value}. Expected a value >= 0.")
