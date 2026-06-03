"""Numeric value object owning safe arithmetic for the calculator tool."""

from __future__ import annotations

from dataclasses import dataclass

MAX_POWER_BASE = 1_000_000
MAX_POWER_EXPONENT = 12


@dataclass(frozen=True)
class Number:
    """A real number that performs guarded arithmetic and renders itself.

    The power guards (base/exponent ceilings) live here so the safety rule
    travels with the operation rather than the caller.

    Example:
        result = Number(2).power(Number(8))  # Number(256.0)
    """

    value: float

    def add(self, other: Number) -> Number:
        return Number(self.value + other.value)

    def subtract(self, other: Number) -> Number:
        return Number(self.value - other.value)

    def multiply(self, other: Number) -> Number:
        return Number(self.value * other.value)

    def divide(self, other: Number) -> Number:
        return Number(self.value / other.value)

    def floor_divide(self, other: Number) -> Number:
        return Number(self.value // other.value)

    def modulo(self, other: Number) -> Number:
        return Number(self.value % other.value)

    def power(self, other: Number) -> Number:
        self._guard_power(other)
        return Number(self.value**other.value)

    def negated(self) -> Number:
        return Number(-self.value)

    def positive(self) -> Number:
        return Number(+self.value)

    def formatted(self) -> str:
        if float(self.value).is_integer():
            return str(int(self.value))

        return str(self.value)

    def _guard_power(self, other: Number) -> None:
        if abs(self.value) > MAX_POWER_BASE:
            raise ValueError(
                f"Power base too large: {self.value}. Expected <= {MAX_POWER_BASE}."
            )

        if abs(other.value) > MAX_POWER_EXPONENT:
            raise ValueError(
                f"Power exponent too large: {other.value}. "
                f"Expected <= {MAX_POWER_EXPONENT}."
            )
