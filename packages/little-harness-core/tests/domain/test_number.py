from __future__ import annotations

import pytest
from little_harness.domain.values.number import Number


class TestNumberArithmetic:
    def test_supports_the_safe_operator_set(self) -> None:
        # Arrange
        six = Number(6)
        two = Number(2)
        seven = Number(7)
        cases: list[tuple[Number, float]] = [
            (six.add(two), 8.0),
            (six.subtract(two), 4.0),
            (six.multiply(two), 12.0),
            (six.divide(two), 3.0),
            (seven.floor_divide(two), 3.0),
            (seven.modulo(two), 1.0),
            (two.power(Number(8)), 256.0),
            (six.negated(), -6.0),
            (six.positive(), 6.0),
        ]

        # Act / Assert
        for actual, expected in cases:
            assert actual.value == expected


class TestNumberPowerGuards:
    def test_rejects_oversized_base(self) -> None:
        # Act / Assert
        with pytest.raises(ValueError, match="Power base too large"):
            Number(1_000_001).power(Number(2))

    def test_rejects_oversized_exponent(self) -> None:
        # Act / Assert
        with pytest.raises(ValueError, match="Power exponent too large"):
            Number(2).power(Number(13))

    def test_allows_values_exactly_at_the_limit(self) -> None:
        # Arrange
        expected_base_result = 1_000_000_000_000.0
        expected_exponent_result = 4096.0

        # Act
        base_result = Number(1_000_000).power(Number(2))
        exponent_result = Number(2).power(Number(12))

        # Assert
        assert base_result.value == expected_base_result
        assert exponent_result.value == expected_exponent_result


class TestNumberZeroDivisionGuards:
    def test_divide_by_zero_raises_value_error(self) -> None:
        # Act / Assert
        with pytest.raises(ValueError, match="Division by zero"):
            Number(6).divide(Number(0))

    def test_floor_divide_by_zero_raises_value_error(self) -> None:
        # Act / Assert
        with pytest.raises(ValueError, match="Division by zero"):
            Number(6).floor_divide(Number(0))

    def test_modulo_by_zero_raises_value_error(self) -> None:
        # Act / Assert
        with pytest.raises(ValueError, match="Division by zero"):
            Number(6).modulo(Number(0))


class TestNumberFormatted:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [(4.0, "4"), (4.25, "4.25")],
    )
    def test_renders_integer_like_floats_without_decimal(
        self,
        value: float,
        expected: str,
    ) -> None:
        # Act / Assert
        assert Number(value).formatted() == expected
