"""Tests for WebFetchTool — HTTP GET via stdlib urllib."""

from __future__ import annotations

from http.client import HTTPMessage
from io import BytesIO
from urllib.error import HTTPError, URLError
from urllib.request import Request

from little_harness.domain.tool_result import ToolRunRequest
from little_harness.domain.values.text_values import ToolInput, ToolName

from little_harness_web_fetch.provider import build as build_web_fetch_tool
from little_harness_web_fetch.web_fetch_tool import WebFetchTool


def fetch_request(raw: str) -> ToolRunRequest:
    return ToolRunRequest(ToolName("web_fetch"), ToolInput(raw))


class FakeUrlOpener:
    """Simulates urllib.request.urlopen for tests."""

    def __init__(self, status: int = 200, body: str = "content") -> None:
        self._status = status
        self._body = body
        self.last_request: Request | None = None
        self.last_timeout: int | None = None

    def open(self, url: Request, *, timeout: int) -> BytesIO:
        self.last_request = url
        self.last_timeout = timeout
        if self._status >= 400:
            msg = HTTPMessage()
            raise HTTPError(url.full_url, self._status, "Error", msg, BytesIO())
        if self._status == 0:
            raise URLError("Connection refused")
        return BytesIO(self._body.encode("utf-8"))


class TestWebFetchToolSpec:
    def test_advertises_the_web_fetch_name(self) -> None:
        spec = WebFetchTool().spec
        assert spec.name == ToolName("web_fetch")

    def test_description_is_not_empty(self) -> None:
        spec = WebFetchTool().spec
        assert spec.description

    def test_input_schema_has_description(self) -> None:
        spec = WebFetchTool().spec
        assert spec.input_schema.description

    def test_has_examples(self) -> None:
        spec = WebFetchTool().spec
        assert not spec.input_schema.examples.is_empty()

    def test_requires_no_approval(self) -> None:
        spec = WebFetchTool().spec
        assert spec.requires_approval is False


class TestWebFetchToolRun:
    def test_fetches_url_content(self) -> None:
        opener = FakeUrlOpener(body="hello world")
        tool = WebFetchTool(urlopen=opener.open)
        request = fetch_request('{"url": "https://example.com"}')
        result = tool.run(request)
        assert result.succeeded is True
        assert "hello world" in result.output.value

    def test_passes_timeout_to_opener(self) -> None:
        opener = FakeUrlOpener()
        tool = WebFetchTool(urlopen=opener.open)
        request = fetch_request('{"url": "https://example.com", "timeout": 15}')
        tool.run(request)
        assert opener.last_timeout == 15

    def test_default_timeout_is_thirty(self) -> None:
        opener = FakeUrlOpener()
        tool = WebFetchTool(urlopen=opener.open)
        request = fetch_request('{"url": "https://example.com"}')
        tool.run(request)
        assert opener.last_timeout == 30

    def test_returns_error_for_http_404(self) -> None:
        opener = FakeUrlOpener(status=404)
        tool = WebFetchTool(urlopen=opener.open)
        request = fetch_request('{"url": "https://example.com/404"}')
        result = tool.run(request)
        assert result.succeeded is False
        assert "404" in result.output.value

    def test_returns_error_for_http_500(self) -> None:
        opener = FakeUrlOpener(status=500)
        tool = WebFetchTool(urlopen=opener.open)
        request = fetch_request('{"url": "https://example.com/500"}')
        result = tool.run(request)
        assert result.succeeded is False
        assert "500" in result.output.value

    def test_returns_error_for_missing_url_field(self) -> None:
        tool = WebFetchTool()
        request = fetch_request('{"format": "text"}')
        result = tool.run(request)
        assert result.succeeded is False
        assert "error" in result.output.value.lower()

    def test_returns_error_for_invalid_json(self) -> None:
        tool = WebFetchTool()
        request = fetch_request("not-json")
        result = tool.run(request)
        assert result.succeeded is False
        assert "error" in result.output.value.lower()

    def test_returns_error_for_non_int_timeout(self) -> None:
        tool = WebFetchTool()
        request = fetch_request('{"url": "https://example.com", "timeout": "slow"}')
        result = tool.run(request)
        assert result.succeeded is False
        assert "timeout" in result.output.value.lower()

    def test_returns_error_on_connection_failure(self) -> None:
        opener = FakeUrlOpener(status=0)
        tool = WebFetchTool(urlopen=opener.open)
        request = fetch_request('{"url": "https://example.com"}')
        result = tool.run(request)
        assert result.succeeded is False
        assert "refused" in result.output.value.lower()

    def test_returns_text_content_by_default(self) -> None:
        opener = FakeUrlOpener(body="<html>hi</html>")
        tool = WebFetchTool(urlopen=opener.open)
        request = fetch_request('{"url": "https://example.com"}')
        result = tool.run(request)
        assert result.output.value == "<html>hi</html>"

    def test_returns_raw_html_when_format_is_html(self) -> None:
        opener = FakeUrlOpener(body="<h1>Title</h1>")
        tool = WebFetchTool(urlopen=opener.open)
        request = fetch_request('{"url": "https://example.com", "format": "html"}')
        result = tool.run(request)
        assert result.output.value == "<h1>Title</h1>"


def test_build_returns_web_fetch_tool() -> None:
    tool = build_web_fetch_tool()
    assert isinstance(tool, WebFetchTool)
    assert tool.spec.name == ToolName("web_fetch")
