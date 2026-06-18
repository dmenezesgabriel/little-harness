"""Latency metrics for a single model completion call.

Separates the fine-grained latency signal (time-to-first-token, generation
throughput) from the coarse `on_model_completed` event, so observers that only
need wall time stay unchanged while latency tooling gets a structured record.
"""

from __future__ import annotations

from dataclasses import dataclass

from little_harness.domain.values.guards import require_non_negative_int
from little_harness.domain.values.numeric_values import ElapsedSeconds


@dataclass(frozen=True)
class ModelCallMetrics:
    """Per-call latency: wall time, time-to-first-token, and tokens generated.

    `time_to_first_token` is None when the call produced no streamed chunks.
    `output_tokens` is approximated by the streamed chunk count (llama.cpp emits
    roughly one token per chunk).

    Example:
        metrics = ModelCallMetrics(ElapsedSeconds(2.0), ElapsedSeconds(0.5), 20)
        metrics.tokens_per_second  # -> 10.0

    """

    elapsed: ElapsedSeconds
    time_to_first_token: ElapsedSeconds | None
    output_tokens: int

    def __post_init__(self) -> None:
        """Validate that the token count is non-negative."""
        require_non_negative_int(self.output_tokens, "OutputTokens")

    @property
    def tokens_per_second(self) -> float:
        """Generation throughput; 0.0 when elapsed is non-positive (avoids /0)."""
        if self.elapsed.value <= 0:
            return 0.0
        return self.output_tokens / self.elapsed.value
