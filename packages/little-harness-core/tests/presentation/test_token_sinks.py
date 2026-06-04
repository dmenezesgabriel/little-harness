from __future__ import annotations

from io import StringIO

from little_harness.application.ports.token_sink import TokenSink
from little_harness.domain.values.text_values import MessageContent
from little_harness.presentation.cli.token_sinks import NullTokenSink, StdoutTokenSink


class TestNullTokenSink:
    def test_conforms_to_the_port_and_discards_chunks(self) -> None:
        # Arrange: the explicit annotation forces a protocol-conformance check.
        sink: TokenSink = NullTokenSink()

        # Act / Assert
        assert sink.emit(MessageContent("ignored")) is None


class TestStdoutTokenSink:
    def test_writes_and_flushes_each_chunk_in_order(self) -> None:
        # Arrange
        buffer = StringIO()
        sink = StdoutTokenSink(buffer)

        # Act
        sink.emit(MessageContent("Hel"))
        sink.emit(MessageContent("lo"))

        # Assert
        assert buffer.getvalue() == "Hello"
