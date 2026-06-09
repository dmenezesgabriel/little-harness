"""Interactive human-in-the-loop permission requester for the CLI.

Shows the pending tool call on an output stream and reads a yes/no answer from
an input stream. Both streams are injected (defaulting to stdout/stdin) so the
prompt is exercised with `StringIO` in tests, mirroring `StdoutTokenSink`.
"""

from __future__ import annotations

import sys
from typing import TextIO

from little_harness.domain.decision import ToolCall

AFFIRMATIVE_ANSWERS = frozenset({"y", "yes"})


class InteractivePermissionRequester:
    """Prompts the operator to approve a sensitive tool call.

    A closed input stream reads as empty, which denies — failing safe rather
    than crashing on EOF.

    Example:
        InteractivePermissionRequester().request_approval(call)

    """

    def __init__(
        self, output: TextIO | None = None, source: TextIO | None = None
    ) -> None:
        """See class docstring for argument descriptions."""
        self._output = output if output is not None else sys.stdout
        self._source = source if source is not None else sys.stdin

    def request_approval(self, call: ToolCall) -> bool:
        """Prompt the operator to approve a tool call and return their answer."""
        self._output.write(
            f"Allow tool {call.tool_name.value!r} to run with input "
            f"{call.tool_input.value!r}? [y/N] "
        )
        self._output.flush()
        answer = self._source.readline()
        return answer.strip().lower() in AFFIRMATIVE_ANSWERS
