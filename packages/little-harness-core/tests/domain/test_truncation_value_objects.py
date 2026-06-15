from __future__ import annotations

import pytest
from little_harness.domain.values.truncation import TruncationConfig, TruncationResult


class TestTruncationConfig:
    def test_default_values(self) -> None:
        config = TruncationConfig()
        assert config.max_lines == 2000
        assert config.max_bytes == 51200

    def test_custom_values(self) -> None:
        config = TruncationConfig(max_lines=100, max_bytes=1024)
        assert config.max_lines == 100
        assert config.max_bytes == 1024

    @pytest.mark.parametrize("value", [0, -1])
    def test_rejects_non_positive_max_lines(self, value: int) -> None:
        with pytest.raises(ValueError, match="MaxLines is not positive"):
            TruncationConfig(max_lines=value)

    @pytest.mark.parametrize("value", [0, -1])
    def test_rejects_non_positive_max_bytes(self, value: int) -> None:
        with pytest.raises(ValueError, match="MaxBytes is not positive"):
            TruncationConfig(max_bytes=value)


class TestTruncationResult:
    def test_not_truncated(self) -> None:
        result = TruncationResult(
            content="hello\nworld",
            truncated=False,
            truncated_by=None,
            total_lines=2,
            total_bytes=11,
            output_lines=2,
            output_bytes=11,
        )
        assert result.content == "hello\nworld"
        assert not result.truncated

    def test_truncated_by_lines(self) -> None:
        result = TruncationResult(
            content="hello",
            truncated=True,
            truncated_by="lines",
            total_lines=5,
            total_bytes=25,
            output_lines=1,
            output_bytes=5,
        )
        assert result.truncated
        assert result.truncated_by == "lines"

    def test_truncated_by_bytes(self) -> None:
        result = TruncationResult(
            content="hello",
            truncated=True,
            truncated_by="bytes",
            total_lines=100,
            total_bytes=5000,
            output_lines=1,
            output_bytes=5,
        )
        assert result.truncated
        assert result.truncated_by == "bytes"
