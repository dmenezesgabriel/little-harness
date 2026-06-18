"""Tests for the ModelCallMetrics value object."""

from __future__ import annotations

import pytest
from little_harness.domain.values.model_call_metrics import ModelCallMetrics
from little_harness.domain.values.numeric_values import ElapsedSeconds


class TestModelCallMetrics:
    def test_tokens_per_second_divides_tokens_by_elapsed(self) -> None:
        metrics = ModelCallMetrics(
            elapsed=ElapsedSeconds(2.0),
            time_to_first_token=ElapsedSeconds(0.5),
            output_tokens=20,
        )

        assert metrics.tokens_per_second == 10.0

    def test_tokens_per_second_is_zero_when_elapsed_is_zero(self) -> None:
        # Guards the divide-by-zero a sub-millisecond call can produce.
        metrics = ModelCallMetrics(
            elapsed=ElapsedSeconds(0.0),
            time_to_first_token=None,
            output_tokens=5,
        )

        assert metrics.tokens_per_second == 0.0

    def test_time_to_first_token_may_be_absent_for_empty_output(self) -> None:
        metrics = ModelCallMetrics(
            elapsed=ElapsedSeconds(1.0),
            time_to_first_token=None,
            output_tokens=0,
        )

        assert metrics.time_to_first_token is None
        assert metrics.output_tokens == 0

    def test_rejects_negative_output_tokens(self) -> None:
        with pytest.raises(ValueError, match="OutputTokens"):
            ModelCallMetrics(
                elapsed=ElapsedSeconds(1.0),
                time_to_first_token=None,
                output_tokens=-1,
            )
