"""Regression tests for bounded upstream downloads and provider requests."""

from collections import deque

import httpx
import pytest

from app.providers.opensubtitles import OpenSubtitlesProvider
from app.providers.subdl import SubdlProvider
from app.providers.subsource import SubsourceProvider
from app.providers.subtitlecat import SubtitlecatProvider
from app.providers.yifysubtitles import YifysubtitlesProvider
from app.utils.http_limits import (
    MAX_HTML_RESPONSE_BYTES,
    MAX_UPSTREAM_DOWNLOAD_BYTES,
    bounded_download_bytes,
    bounded_fetch_text,
)


@pytest.fixture(autouse=True)
def _resolve_test_provider_hosts_as_public(monkeypatch):
    from app.utils import http_limits

    monkeypatch.setattr(
        http_limits.socket,
        "getaddrinfo",
        lambda host, port, **kwargs: [(2, 1, 6, "", ("93.184.216.34", port or 443))],
    )


class MockResponse:
    def __init__(self, status_code=200, chunks=(), headers=None, text_encoding="utf-8"):
        self.status_code = status_code
        self._chunks = list(chunks)
        self.headers = dict(headers or {})
        if "content-length" not in self.headers and self._chunks:
            self.headers["content-length"] = str(sum(map(len, self._chunks)))
        self.encoding = text_encoding
        self.iter_entered = False
        self.chunks_yielded = 0
        self.chunk_sizes = []

    async def aiter_bytes(self, chunk_size=None):
        self.iter_entered = True
        self.chunk_sizes.append(chunk_size)
        for chunk in self._chunks:
            self.chunks_yielded += 1
            yield chunk


class StreamContext:
    def __init__(self, response=None, enter_error=None):
        self.response = response
        self.enter_error = enter_error

    async def __aenter__(self):
        if self.enter_error:
            raise self.enter_error
        return self.response

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class MockHttpClient:
    """httpx-compatible stream mock; stream() is sync and returns an async CM."""

    def __init__(self, responses=()):
        self.responses = deque(responses)
        self.stream_count = 0
        self.stream_calls = []
        self.get_count = 0
        self.post_count = 0
        self.post_calls = []

    def stream(self, method, url, **kwargs):
        self.stream_count += 1
        self.stream_calls.append({"method": method, "url": str(url), "kwargs": kwargs})
        if not self.responses:
            raise AssertionError(f"Unexpected stream request: {method} {url}")
        next_response = self.responses.popleft()
        if isinstance(next_response, BaseException):
            return StreamContext(enter_error=next_response)
        if callable(next_response):
            next_response = next_response(method, url, kwargs)
        return next_response if isinstance(next_response, StreamContext) else StreamContext(next_response)

    async def get(self, *args, **kwargs):
        self.get_count += 1
        raise AssertionError("Download must not use a GET preflight")

    async def post(self, url, **kwargs):
        self.post_count += 1
        self.post_calls.append({"url": url, "kwargs": kwargs})
        raise AssertionError("Unexpected POST request")


def response(data: bytes, *, status=200, headers=None):
    actual_headers = dict(headers or {})
    actual_headers.setdefault("content-length", str(len(data)))
    return MockResponse(status, [data], actual_headers)


@pytest.mark.asyncio
async def test_content_length_over_limit_rejected_before_iteration():
    resp = MockResponse(200, [b"not-read"], {"content-length": str(MAX_UPSTREAM_DOWNLOAD_BYTES + 1)})
    client = MockHttpClient([resp])
    status, data = await bounded_download_bytes(client, "https://example.test/archive")
    assert (status, data) == (200, b"")
    assert resp.iter_entered is False
    assert resp.chunks_yielded == 0
    assert client.stream_count == 1


@pytest.mark.asyncio
async def test_streamed_overflow_without_large_content_length_aborts_after_chunks():
    """No oversized Content-Length: cumulative body size alone triggers rejection."""
    first = b"a" * (MAX_UPSTREAM_DOWNLOAD_BYTES // 2)
    second = b"b" * (MAX_UPSTREAM_DOWNLOAD_BYTES // 2)
    third = b"c" * 1
    resp = MockResponse(200, [first, second, third], {"content-length": str(MAX_UPSTREAM_DOWNLOAD_BYTES)})
    client = MockHttpClient([resp])
    status, data = await bounded_download_bytes(client, "https://example.test/chunked")
    assert (status, data) == (200, b"")
    assert resp.iter_entered is True
    assert resp.chunks_yielded == 3
    assert resp.chunk_sizes == [64 * 1024]
    assert int(resp.headers["content-length"]) <= MAX_UPSTREAM_DOWNLOAD_BYTES


@pytest.mark.asyncio
async def test_valid_bounded_download_is_one_stream_and_no_get_preflight():
    payload = b"subtitle bytes"
    client = MockHttpClient([response(payload)])
    status, data = await bounded_download_bytes(client, "https://example.test/subtitle")
    assert (status, data) == (200, payload)
    assert client.stream_count == 1
    assert client.get_count == 0


@pytest.mark.asyncio
async def test_non_200_body_is_not_iterated():
    resp = MockResponse(404, [b"error page"], {"content-length": "10"})
    client = MockHttpClient([resp])
    status, data = await bounded_download_bytes(client, "https://example.test/missing")
    assert (status, data) == (404, b"")
    assert resp.iter_entered is False


@pytest.mark.asyncio
async def test_timeout_exception_is_raised_by_context_enter():
    client = MockHttpClient([httpx.TimeoutException("timed out")])
    status, data = await bounded_download_bytes(client, "https://example.test/slow")
    assert (status, data) == (0, b"")
    assert client.stream_count == 1


@pytest.mark.asyncio
async def test_request_error_exception_is_raised_by_context_enter():
    client = MockHttpClient([httpx.RequestError("connection failed")])
    status, data = await bounded_download_bytes(client, "https://example.test/unreachable")
    assert (status, data) == (0, b"")
    assert client.stream_count == 1


@pytest.mark.asyncio
async def test_bounded_html_large_body_rejected_and_normal_html_decodes():
    large = MockResponse(
        200,
        [b"x" * (MAX_HTML_RESPONSE_BYTES + 1)],
        {"content-length": str(MAX_HTML_RESPONSE_BYTES)},
    )
    status, text = await bounded_fetch_text(MockHttpClient([large]), "https://example.test/large")
    assert (status, text) == (200, "")
    assert large.iter_entered is True
    assert large.chunk_sizes == [64 * 1024]

    small = response(b"<html>ok</html>")
    status, text = await bounded_fetch_text(MockHttpClient([small]), "https://example.test/page")
    assert (status, text) == (200, "<html>ok</html>")


@pytest.mark.asyncio
async def test_yify_and_subtitlecat_oversized_html_search_fail_gracefully():
    too_large = MockResponse(
        200,
        [b"x" * (MAX_HTML_RESPONSE_BYTES + 1)],
        {"content-length": str(MAX_HTML_RESPONSE_BYTES + 1)},
    )
    yify_client = MockHttpClient([too_large])
    assert await YifysubtitlesProvider(yify_client).search_subtitles(
        "tt0111161", languages=["ara"]
    ) == []
    assert yify_client.stream_count == 1
    assert too_large.iter_entered is False

    cat_response = MockResponse(
        200,
        [b"x" * (MAX_HTML_RESPONSE_BYTES + 1)],
        {"content-length": str(MAX_HTML_RESPONSE_BYTES + 1)},
    )
    cat_client = MockHttpClient([cat_response])
    assert await SubtitlecatProvider(cat_client).search_subtitles(
        "tt0111161", title="A Movie", year=2024, languages=["ara"]
    ) == []
    assert cat_client.stream_count == 1
    assert cat_response.iter_entered is False


@pytest.mark.asyncio
async def test_subdl_download_is_bounded_and_one_request():
    payload = b"subdl archive"
    client = MockHttpClient([response(payload)])
    result = await SubdlProvider(client).download_archive(
        "https://dl.subdl.com/path/file.zip", api_key="subdl-secret"
    )
    assert result == payload
    assert client.stream_count == 1
    assert client.get_count == 0
    assert client.stream_calls[0]["kwargs"]["headers"]["x-api-key"] == "subdl-secret"


@pytest.mark.asyncio
async def test_subsource_direct_binary_download_is_bounded_and_one_request():
    payload = b"direct subsource bytes"
    client = MockHttpClient([response(payload)])
    result = await SubsourceProvider(client).download_archive("123", api_key="ss-key")
    assert result == payload
    assert client.stream_count == 1
    assert client.get_count == 0
    assert client.stream_calls[0]["kwargs"]["headers"]["X-API-Key"] == "ss-key"


@pytest.mark.asyncio
async def test_subsource_cdn_flow_bounds_both_requests_and_strips_secrets():
    payload = b"subtitle bytes from CDN"
    api_json = b'{"downloadUrl":"https://cdn.subsource.net/signed?token=private"}'
    api_response = response(api_json)
    cdn_response = response(payload)
    client = MockHttpClient([api_response, cdn_response])
    result = await SubsourceProvider(client).download_archive("456", api_key="ss-private")

    assert result == payload
    assert client.stream_count == 2
    assert client.get_count == 0
    first_headers = client.stream_calls[0]["kwargs"]["headers"]
    cdn_headers = client.stream_calls[1]["kwargs"]["headers"]
    assert first_headers["X-API-Key"] == "ss-private"
    assert "X-API-Key" not in cdn_headers
    assert "Authorization" not in cdn_headers
    assert cdn_headers["User-Agent"] == first_headers["User-Agent"]
    assert cdn_response.iter_entered is True


@pytest.mark.asyncio
async def test_cross_origin_redirect_strips_subdl_api_key_but_follows_redirect():
    redirect = MockResponse(302, [], {"location": "https://cdn.subdl.com/file.zip"})
    payload = b"redirected file"
    captured = []

    def capture_second(method, url, kwargs):
        captured.append((url, kwargs["headers"]))
        return response(payload)

    client = MockHttpClient([redirect, capture_second])
    result = await SubdlProvider(client).download_archive(
        "https://api.subdl.com/file?sig=secret", api_key="subdl-key"
    )
    assert result == payload
    assert client.stream_count == 2
    assert captured[0][0] == "https://cdn.subdl.com/file.zip"
    assert "x-api-key" not in {name.lower() for name in captured[0][1]}
    assert client.stream_calls[1]["kwargs"].get("params") is None


@pytest.mark.asyncio
async def test_subsource_authenticated_redirect_to_cdn_is_rejected_before_credentials_forward():
    redirect = MockResponse(302, [], {"location": "https://media.cloudfront.net/subtitle.srt"})
    client = MockHttpClient([redirect])
    result = await SubsourceProvider(client).download_archive("sub-redirect", api_key="secret")
    assert result is None
    assert client.stream_count == 1
    assert redirect.iter_entered is False


@pytest.mark.asyncio
async def test_opensubtitles_direct_download_is_bounded_one_stream_request():
    payload = b"opensubtitles data"
    client = MockHttpClient([response(payload)])
    result = await OpenSubtitlesProvider(client).download_archive(
        "https://dl.opensubtitles.com/file?token=signed"
    )
    assert result == payload
    assert client.stream_count == 1
    assert client.get_count == 0
    assert client.post_count == 0


@pytest.mark.asyncio
async def test_yify_direct_download_is_bounded_one_request():
    payload = b"yify zip"
    client = MockHttpClient([response(payload)])
    result = await YifysubtitlesProvider(client).download_archive(
        "https://yifysubtitles.ch/subtitle/movie.zip"
    )
    assert result == payload
    assert client.stream_count == 1
    assert client.get_count == 0


@pytest.mark.asyncio
async def test_subtitlecat_direct_download_is_bounded_one_request():
    payload = b"1\n00:00:01,000 --> 00:00:02,000\nHello\n"
    client = MockHttpClient([response(payload)])
    result = await SubtitlecatProvider(client).download_archive(
        "https://www.subtitlecat.com/subs/1/file-ar.srt"
    )
    assert result == payload
    assert client.stream_count == 1
    assert client.get_count == 0
