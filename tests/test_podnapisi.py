"""Bounded Podnapisi search/download tests for AutoSync references."""

from collections import deque

import httpx
import pytest

from app.providers.podnapisi import PodnapisiProvider

HTML = """<table><tr data-pid="111">
<td><abbr title="English">EN</abbr></td>
<td><span class="release">Show.S01E01.1080p.WEB-DL.x264-GRP</span></td>
</tr></table>"""


class Response:
    def __init__(self, status=200, chunks=(), headers=None):
        self.status_code = status
        self.headers = dict(headers or {})
        self.chunks = list(chunks)
        self.encoding = "utf-8"
        self.iter_entered = False
        self.yielded = 0
        self.chunk_sizes = []

    async def aiter_bytes(self, chunk_size=None):
        self.iter_entered = True
        self.chunk_sizes.append(chunk_size)
        for chunk in self.chunks:
            self.yielded += 1
            yield chunk


class Context:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error

    async def __aenter__(self):
        if self.error:
            raise self.error
        return self.response

    async def __aexit__(self, exc_type, exc, tb):
        return False


class Client:
    def __init__(self, values):
        self.values = deque(values)
        self.calls = []

    def stream(self, method, url, **kwargs):
        self.calls.append((method, str(url), kwargs))
        value = self.values.popleft()
        return value if isinstance(value, Context) else Context(value)


@pytest.fixture(autouse=True)
def fake_public_dns(monkeypatch):
    from app.utils import http_limits

    monkeypatch.setattr(
        http_limits.socket,
        "getaddrinfo",
        lambda host, port, **kwargs: [(2, 1, 6, "", ("93.184.216.34", port or 443))],
    )


@pytest.mark.asyncio
async def test_podnapisi_normal_search_and_season_episode_params():
    response = Response(200, [HTML.encode()], {"content-length": str(len(HTML))})
    client = Client([response])
    provider = PodnapisiProvider(client)
    releases = await provider.search_subtitles(
        "tt12345", is_series=True, season=1, episode=1,
        title="Show", year=2024, languages=["eng"],
    )
    assert len(releases) == 1
    assert releases[0].provider == "podnapisi"
    assert releases[0].lang == "eng"
    assert releases[0].download_url == "https://www.podnapisi.net/subtitles/111/download"
    assert client.calls[0][2]["params"] == {
        "keywords": "Show", "movie_type": "tv-series", "language": "en",
        "seasons": "1", "episodes": "1", "year": "2024", "imdb": "tt12345",
    }
    assert response.iter_entered
    assert response.chunk_sizes == [64 * 1024]


@pytest.mark.asyncio
async def test_podnapisi_rejects_html_content_length_overflow_before_iteration():
    from app.utils.http_limits import MAX_HTML_RESPONSE_BYTES

    response = Response(
        200, [b"unread"], {"content-length": str(MAX_HTML_RESPONSE_BYTES + 1)}
    )
    releases = await PodnapisiProvider(Client([response])).search_subtitles("tt1", title="Show")
    assert releases == []
    assert response.iter_entered is False


@pytest.mark.asyncio
async def test_podnapisi_rejects_streamed_html_overflow():
    from app.utils.http_limits import MAX_HTML_RESPONSE_BYTES

    response = Response(
        200,
        [b"a" * (MAX_HTML_RESPONSE_BYTES // 2), b"b" * (MAX_HTML_RESPONSE_BYTES // 2), b"c"],
        {"content-length": str(MAX_HTML_RESPONSE_BYTES)},
    )
    releases = await PodnapisiProvider(Client([response])).search_subtitles("tt1", title="Show")
    assert releases == []
    assert response.iter_entered is True
    assert response.yielded == 3
    assert int(response.headers["content-length"]) <= MAX_HTML_RESPONSE_BYTES


@pytest.mark.asyncio
async def test_podnapisi_normal_bounded_archive_download():
    payload = b"PK\x03\x04" + b"archive-data"
    response = Response(200, [payload], {"content-length": str(len(payload))})
    result = await PodnapisiProvider(Client([response])).download_archive("111")
    assert result == payload


@pytest.mark.asyncio
async def test_podnapisi_rejects_binary_content_length_overflow():
    from app.utils.http_limits import MAX_UPSTREAM_DOWNLOAD_BYTES

    response = Response(
        200, [b"unread"], {"content-length": str(MAX_UPSTREAM_DOWNLOAD_BYTES + 1)}
    )
    result = await PodnapisiProvider(Client([response])).download_archive("111")
    assert result is None
    assert response.iter_entered is False


@pytest.mark.asyncio
async def test_podnapisi_rejects_streamed_binary_overflow():
    from app.utils.http_limits import MAX_UPSTREAM_DOWNLOAD_BYTES

    response = Response(
        200,
        [b"a" * (MAX_UPSTREAM_DOWNLOAD_BYTES // 2), b"b" * (MAX_UPSTREAM_DOWNLOAD_BYTES // 2), b"c"],
        {"content-length": str(MAX_UPSTREAM_DOWNLOAD_BYTES)},
    )
    result = await PodnapisiProvider(Client([response])).download_archive("111")
    assert result is None
    assert response.iter_entered is True
    assert response.yielded == 3


@pytest.mark.parametrize(
    "error",
    [httpx.TimeoutException("timed out"), httpx.RequestError("upstream error")],
)
@pytest.mark.asyncio
async def test_podnapisi_timeout_and_request_error_fail_safely(error):
    provider = PodnapisiProvider(Client([Context(error=error)]))
    assert await provider.download_archive("111") is None


@pytest.mark.asyncio
async def test_podnapisi_non_200_body_is_not_read():
    response = Response(503, [b"error body"], {"content-length": "11"})
    result = await PodnapisiProvider(Client([response])).download_archive("111")
    assert result is None
    assert response.iter_entered is False


@pytest.mark.parametrize(
    "location",
    ["https://evil.example/file.zip", "https://127.0.0.1/latest/meta-data/"],
)
@pytest.mark.asyncio
async def test_podnapisi_rejects_unapproved_redirect_host(location):
    redirect = Response(302, [], {"location": location})
    client = Client([redirect])
    assert await PodnapisiProvider(client).download_archive("111") is None
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_podnapisi_rejects_unapproved_initial_url_and_non_numeric_relative_ref():
    client = Client([])
    provider = PodnapisiProvider(client)
    assert await provider.download_archive("https://evil.example/subtitles/111/download") is None
    assert await provider.download_archive("../../internal") is None
    assert client.calls == []


@pytest.mark.asyncio
async def test_podnapisi_never_logs_signed_url_or_token(caplog):
    signed_url = "https://www.podnapisi.net/subtitles/111/download?token=DO_NOT_LOG"
    response = Response(503, [], {})
    client = Client([response])
    with caplog.at_level("WARNING"):
        assert await PodnapisiProvider(client).download_archive(signed_url) is None
    assert "DO_NOT_LOG" not in caplog.text
    assert "/subtitles/111/download" not in caplog.text
