"""Truncator that keeps the last N lines of tool output."""

from __future__ import annotations

from little_harness.domain.values.truncation import TruncationConfig, TruncationResult

_UTF8_CONTINUATION_MASK = 0xC0
_UTF8_CONTINUATION_MARKER = 0x80


class TailTruncator:
    """Truncate content from the tail, keeping the last N lines/bytes.

    Suitable for bash output where errors and final results are at the end.
    May return a partial last line if that single line exceeds the byte limit.
    """

    def truncate(self, content: str, config: TruncationConfig) -> TruncationResult:
        """Truncate content, keeping tail lines."""
        total_bytes = len(content.encode("utf-8"))
        lines = self._split_lines(content)
        total_lines = len(lines)

        if total_lines <= config.max_lines and total_bytes <= config.max_bytes:
            return TruncationResult(
                content=content,
                truncated=False,
                truncated_by=None,
                total_lines=total_lines,
                total_bytes=total_bytes,
                output_lines=total_lines,
                output_bytes=total_bytes,
            )

        output_lines, truncated_by = self._collect_tail(lines, config)

        output_content = "\n".join(output_lines)
        final_output_bytes = len(output_content.encode("utf-8"))

        return TruncationResult(
            content=output_content,
            truncated=True,
            truncated_by=truncated_by,
            total_lines=total_lines,
            total_bytes=total_bytes,
            output_lines=len(output_lines),
            output_bytes=final_output_bytes,
        )

    @staticmethod
    def _collect_tail(
        lines: list[str], config: TruncationConfig
    ) -> tuple[list[str], str]:
        output_lines: list[str] = []
        output_bytes_count = 0
        truncated_by: str = "lines"

        for i in range(len(lines) - 1, -1, -1):
            if len(output_lines) >= config.max_lines:
                break

            line = lines[i]
            line_bytes = len(line.encode("utf-8"))

            if (
                output_bytes_count + line_bytes + (1 if output_lines else 0)
                > config.max_bytes
            ):
                truncated_by = "bytes"
                if not output_lines:
                    truncated_line = TailTruncator._truncate_string_from_end(
                        line, config.max_bytes
                    )
                    output_lines.insert(0, truncated_line)
                    output_bytes_count = len(truncated_line.encode("utf-8"))
                break

            output_lines.insert(0, line)
            output_bytes_count += line_bytes + (1 if len(output_lines) > 1 else 0)

        return output_lines, truncated_by

    @staticmethod
    def _truncate_string_from_end(text: str, max_bytes: int) -> str:
        encoded = text.encode("utf-8")
        if len(encoded) <= max_bytes:
            return text
        start = len(encoded) - max_bytes
        while (
            start < len(encoded)
            and (encoded[start] & _UTF8_CONTINUATION_MASK) == _UTF8_CONTINUATION_MARKER
        ):
            start += 1
        return encoded[start:].decode("utf-8")

    @staticmethod
    def _split_lines(content: str) -> list[str]:
        if content == "":
            return []
        lines = content.split("\n")
        if content.endswith("\n"):
            lines.pop()
        return lines
