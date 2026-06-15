from __future__ import annotations

from little_harness.domain.values.truncation import TruncationConfig
from little_harness.infrastructure.truncation.head_truncator import HeadTruncator
from little_harness.infrastructure.truncation.tail_truncator import TailTruncator


class TestHeadTruncator:
    def test_no_truncation_needed(self) -> None:
        truncator = HeadTruncator()
        config = TruncationConfig(max_lines=100, max_bytes=10_000)
        result = truncator.truncate("hello\nworld", config)
        assert result.content == "hello\nworld"
        assert not result.truncated

    def test_truncates_by_lines(self) -> None:
        truncator = HeadTruncator()
        config = TruncationConfig(max_lines=2, max_bytes=10_000)
        content = "line1\nline2\nline3\nline4"
        result = truncator.truncate(content, config)
        assert result.content == "line1\nline2"
        assert result.truncated
        assert result.truncated_by == "lines"
        assert result.total_lines == 4
        assert result.output_lines == 2

    def test_truncates_by_bytes(self) -> None:
        truncator = HeadTruncator()
        # Each line is 6 bytes ("lineX\n"), but we count without trailing newline
        # "line1\nline2\n" = 12 bytes; set limit to 10 to cut after one line
        config = TruncationConfig(max_lines=100, max_bytes=10)
        content = "line1\nline2\nline3"
        result = truncator.truncate(content, config)
        assert result.content == "line1"
        assert result.truncated
        assert result.truncated_by == "bytes"
        assert result.output_lines == 1

    def test_first_line_exceeds_byte_limit(self) -> None:
        truncator = HeadTruncator()
        config = TruncationConfig(max_lines=100, max_bytes=5)
        content = "hello world\nnext line"
        result = truncator.truncate(content, config)
        assert result.content == ""
        assert result.truncated
        assert result.total_lines == 2

    def test_empty_content(self) -> None:
        truncator = HeadTruncator()
        config = TruncationConfig()
        result = truncator.truncate("", config)
        assert result.content == ""
        assert not result.truncated

    def test_single_line_fits(self) -> None:
        truncator = HeadTruncator()
        config = TruncationConfig(max_lines=1, max_bytes=10_000)
        result = truncator.truncate("only line", config)
        assert result.content == "only line"
        assert not result.truncated

    def test_preserves_unicode(self) -> None:
        truncator = HeadTruncator()
        config = TruncationConfig(max_lines=5, max_bytes=100)
        content = "café\nrésumé\n👋 hello"
        result = truncator.truncate(content, config)
        assert result.content == content
        assert not result.truncated

    def test_truncates_by_bytes_with_unicode(self) -> None:
        truncator = HeadTruncator()
        # Each "é" is 2 bytes in UTF-8; "café" = 5 bytes
        config = TruncationConfig(max_lines=100, max_bytes=6)
        content = "café\nworld"
        result = truncator.truncate(content, config)
        assert result.content == "café"
        assert result.truncated
        assert result.truncated_by == "bytes"


class TestTailTruncator:
    def test_no_truncation_needed(self) -> None:
        truncator = TailTruncator()
        config = TruncationConfig(max_lines=100, max_bytes=10_000)
        result = truncator.truncate("hello\nworld", config)
        assert result.content == "hello\nworld"
        assert not result.truncated

    def test_keeps_last_lines(self) -> None:
        truncator = TailTruncator()
        config = TruncationConfig(max_lines=2, max_bytes=10_000)
        content = "line1\nline2\nline3\nline4"
        result = truncator.truncate(content, config)
        assert result.content == "line3\nline4"
        assert result.truncated
        assert result.truncated_by == "lines"
        assert result.total_lines == 4
        assert result.output_lines == 2

    def test_truncates_by_bytes_from_tail(self) -> None:
        truncator = TailTruncator()
        config = TruncationConfig(max_lines=100, max_bytes=10)
        # "line3\nline4" = 11 bytes, too much; "line4" = 5 bytes, fits
        content = "line1\nline2\nline3\nline4"
        result = truncator.truncate(content, config)
        assert result.content == "line4"
        assert result.truncated
        assert result.truncated_by == "bytes"

    def test_empty_content(self) -> None:
        truncator = TailTruncator()
        config = TruncationConfig()
        result = truncator.truncate("", config)
        assert result.content == ""
        assert not result.truncated

    def test_single_line_fits(self) -> None:
        truncator = TailTruncator()
        config = TruncationConfig(max_lines=1, max_bytes=10_000)
        result = truncator.truncate("only line", config)
        assert result.content == "only line"
        assert not result.truncated

    def test_partial_last_line_when_first_exceeds(self) -> None:
        truncator = TailTruncator()
        config = TruncationConfig(max_lines=100, max_bytes=5)
        content = "line1\nline2\nhello world"
        result = truncator.truncate(content, config)
        assert result.content == "world"
        assert result.truncated
