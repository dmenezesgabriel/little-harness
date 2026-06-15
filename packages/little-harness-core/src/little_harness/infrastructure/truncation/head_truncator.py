"""Truncator that keeps the first N lines of tool output."""

from __future__ import annotations

from little_harness.domain.values.truncation import TruncationConfig, TruncationResult


class HeadTruncator:
    """Truncate content from the head, keeping the first N lines/bytes.

    Never returns partial lines. If the first line alone exceeds the byte limit,
    returns empty content. Counts bytes as UTF-8 encoded size.
    """

    def truncate(self, content: str, config: TruncationConfig) -> TruncationResult:
        """Truncate content, keeping head lines. Never partial lines."""
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

        first_line_bytes = len(lines[0].encode("utf-8"))
        if first_line_bytes > config.max_bytes:
            return TruncationResult(
                content="",
                truncated=True,
                truncated_by="bytes",
                total_lines=total_lines,
                total_bytes=total_bytes,
                output_lines=0,
                output_bytes=0,
            )

        output_lines, truncated_by = self._collect_head(lines, config)

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
    def _collect_head(
        lines: list[str], config: TruncationConfig
    ) -> tuple[list[str], str]:
        output_lines: list[str] = []
        output_bytes_count = 0
        truncated_by: str = "lines"

        for i, line in enumerate(lines):
            if i >= config.max_lines:
                truncated_by = "lines"
                break

            line_bytes = len(line.encode("utf-8"))
            if (
                output_bytes_count + line_bytes + (1 if output_lines else 0)
                > config.max_bytes
            ):
                truncated_by = "bytes"
                break

            output_lines.append(line)
            output_bytes_count += line_bytes + (1 if len(output_lines) > 1 else 0)

        return output_lines, truncated_by

    @staticmethod
    def _split_lines(content: str) -> list[str]:
        if content == "":
            return []
        lines = content.split("\n")
        if content.endswith("\n"):
            lines.pop()
        return lines
