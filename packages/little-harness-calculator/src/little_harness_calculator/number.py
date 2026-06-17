"""Numeric value object owning safe arithmetic for the calculator tool."""

from __future__ import annotations

from dataclasses import dataclass

MAX_POWER_BASE = 1_000_000
MAX_POWER_EXPONENT = 12


@dataclass(frozen=True)
class Number:
    """A real number that performs guarded arithmetic and renders itself.

    Example:
        result = Number(2).power(Number(8))  # Number(256.0)
    """

    value: float

    def add(self, other: Number) -> Number:
        """Return the sum of this number and `other`."""
        return Number(self.value + other.value)

    def subtract(self, other: Number) -> Number:
        """Return the difference between this number and `other`."""
        return Number(self.value - other.value)

    def multiply(self, other: Number) -> Number:
        """Return the product of this number and `other`."""
        return Number(self.value * other.value)

    def divide(self, other: Number) -> Number:
        """Return the quotient of this number divided by `other`."""
        other._guard_nonzero_divisor(self)
        return Number(self.value / other.value)

    def floor_divide(self, other: Number) -> Number:
        """Return the floor of the quotient of this number divided by `other`."""
        other._guard_nonzero_divisor(self)
        return Number(self.value // other.value)

    def modulo(self, other: Number) -> Number:
        """Return the remainder of this number divided by `other`."""
        other._guard_nonzero_divisor(self)
        return Number(self.value % other.value)

    def power(self, other: Number) -> Number:
        """Return this number raised to the power of `other`."""
        self._guard_power(other)
        return Number(self.value**other.value)

    def negated(self) -> Number:
        """Return the additive inverse of this number."""
        return Number(-self.value)

    def positive(self) -> Number:
        """Return an explicit positive form of this number."""
        return Number(+self.value)

    def formatted(self) -> str:
        """Render the number as an integer string when whole, else decimal."""
        if float(self.value).is_integer():
            return str(int(self.value))

        return str(self.value)

    def _guard_nonzero_divisor(self, dividend: Number) -> None:
        if self.value != 0:
            return

        raise ValueError(
            f"Division by zero: {dividend.value}/0. Expected a non-zero divisor."
        )

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
