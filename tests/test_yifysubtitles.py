"""Unit tests for the YIFYSubtitles (yifysubtitles.ch) provider."""


import pytest

from app.providers.yifysubtitles import YifysubtitlesProvider

MOVIE_HTML = """
<html><body>
<table class="table">
<tbody>
<tr data-id="125583">
  <td class="rating-cell"><span class="label">0</span></td>
  <td class="flag-cell"><span class="flag flag-sa"></span><span class="sub-lang">Arabic</span></td>
  <td><a href="/subtitles/the-shawshank-redemption-1994-arabic-yify-125583"><span class="text-muted">subtitle</span> The.Shawshank.Redemption.1994.1080p.BluRay.x264.[YTS.AG]</a></td>
  <td class="other-cell"></td>
  <td class="uploader-cell"><a href="/user/sub">sub</a></td>
</tr>
<tr data-id="125590">
  <td class="rating-cell"><span class="label">5</span></td>
  <td class="flag-cell"><span class="flag flag-us"></span><span class="sub-lang">English</span></td>
  <td><a href="/subtitles/the-shawshank-redemption-1994-english-yify-125590">The.Shawshank.Redemption.1994.720p.WEB-DL</a></td>
  <td class="other-cell"></td>
  <td class="uploader-cell"><a href="/user/joe">joe</a></td>
</tr>
</tbody>
</table>
</body></html>
"""


class MockResponse:
    """A mock HTTP response that works with bounded_fetch_text."""
    def __init__(self, status_code: int = 200, content: bytes = b"", text: str = ""):
        if text:
            content = text.encode("utf-8")
        self.status_code = status_code
        self.headers = {"content-length": str(len(content))}
        self.encoding = "utf-8"
        self.content = content
        self._content = content

    async def aiter_bytes(self, chunk_size=None):
        yield self._content


class MockStreamCM:
    """An async context manager that returns a mock response."""
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *args):
        pass


class MockHttpClient:
    """A mock HTTP client that properly implements stream() as a sync method returning async CM."""

    def __init__(self, stream_return_value=None, stream_side_effect=None):
        self._stream_return_value = stream_return_value
        self._stream_side_effect = stream_side_effect
        self.call_count = {"stream": 0}

    def stream(self, method, url, **kwargs):
        if self._stream_side_effect:
            return self._stream_side_effect(method, url, **kwargs)
        return self._stream_return_value


@pytest.mark.asyncio
async def test_yifysubtitles_movie_parsing_filters_language():
    """Arabic-only request keeps just the Arabic row and maps metadata correctly."""
    from app.providers.yifysubtitles import YifysubtitlesProvider

    stream_cm = MockStreamCM(MockResponse(200, text=MOVIE_HTML))
    provider = YifysubtitlesProvider(MockHttpClient(stream_return_value=stream_cm))
    subs = await provider.search_subtitles(
        imdb_id="tt0111161",
        is_series=False,
        languages=["ara"],
    )

    assert len(subs) == 1
    sub = subs[0]
    assert sub.provider == "yifysubtitles"
    assert sub.lang == "ara"
    assert sub.release_name == "The.Shawshank.Redemption.1994.1080p.BluRay.x264.[YTS.AG]"
    assert sub.download_url == (
        "https://yifysubtitles.ch/subtitle/"
        "the-shawshank-redemption-1994-arabic-yify-125583.zip"
    )
    assert sub.uploader == "sub"


@pytest.mark.asyncio
async def test_yifysubtitles_multi_release_uses_first_name():
    """Rows listing several releases separated by <br /> keep only the first name."""
    html = """
    <table><tbody>
    <tr data-id="357712">
      <td class="rating-cell"><span class="label">0</span></td>
      <td class="flag-cell"><span class="flag flag-sa"></span><span class="sub-lang">Arabic</span></td>
      <td><a href="/subtitles/the-shawshank-redemption-1994-arabic-yify-357712"><span class="text-muted">subtitle</span> First.Release.1080p.BluRay<br />
Second.Release.720p.BRRip<br />
Third.Release.x264</a></td>
      <td class="other-cell"></td>
      <td class="uploader-cell"><a href="/user/x">x</a></td>
    </tr>
    </tbody></table>
    """
    stream_cm = MockStreamCM(MockResponse(200, text=html))
    provider = YifysubtitlesProvider(MockHttpClient(stream_return_value=stream_cm))
    subs = await provider.search_subtitles(imdb_id="tt0111161", is_series=False, languages=["ara"])

    assert len(subs) == 1
    assert subs[0].release_name == "First.Release.1080p.BluRay"
    assert "\n" not in subs[0].release_name


@pytest.mark.asyncio
async def test_yifysubtitles_multi_language():
    """Requesting Arabic + English returns both rows."""
    from app.providers.yifysubtitles import YifysubtitlesProvider

    stream_cm = MockStreamCM(MockResponse(200, text=MOVIE_HTML))
    provider = YifysubtitlesProvider(MockHttpClient(stream_return_value=stream_cm))
    subs = await provider.search_subtitles(
        imdb_id="tt0111161",
        is_series=False,
        languages=["ara", "eng"],
    )

    assert {s.lang for s in subs} == {"ara", "eng"}


@pytest.mark.asyncio
async def test_yifysubtitles_series_unsupported():
    """YIFYSubtitles is movies-only: series must short-circuit without requests."""
    mock_client = MockHttpClient()
    provider = YifysubtitlesProvider(mock_client)

    subs = await provider.search_subtitles(imdb_id="tt0903747", is_series=True)
    assert subs == []
    assert mock_client.call_count["stream"] == 0


@pytest.mark.asyncio
async def test_yifysubtitles_page_not_found():
    """A 404 / 'Page not found' listing returns no results."""
    stream_cm = MockStreamCM(MockResponse(404, text="Page not found"))
    mock_client = MockHttpClient(stream_return_value=stream_cm)
    provider = YifysubtitlesProvider(mock_client)

    subs = await provider.search_subtitles(imdb_id="tt0000000", is_series=False)
    assert subs == []


@pytest.mark.asyncio
async def test_yifysubtitles_download_zip():
    """download_archive returns raw zip bytes on success."""
    stream_cm = MockStreamCM(MockResponse(200, content=b"PK\x03\x04zipdata"))
    mock_client = MockHttpClient(stream_return_value=stream_cm)
    provider = YifysubtitlesProvider(mock_client)
    data = await provider.download_archive(
        "https://yifysubtitles.ch/subtitle/the-shawshank-redemption-1994-arabic-yify-125583.zip"
    )
    assert data == b"PK\x03\x04zipdata"


@pytest.mark.asyncio
async def test_yifysubtitles_download_cloudflare_warmup():
    """On a Cloudflare 403 the provider warms up the session once and retries."""
    mock_client = MockHttpClient()
    zip_attempts = {"count": 0}

    def stream_side_effect(method, url, **kwargs):
        if url.endswith(".zip"):
            if zip_attempts["count"] == 0:
                zip_attempts["count"] = 1
                return MockStreamCM(MockResponse(403, text="<cf challenge>"))
            return MockStreamCM(MockResponse(200, content=b"PK\x03\x04retried"))
        return MockStreamCM(MockResponse(200, text="<home>"))

    mock_client = MockHttpClient()
    mock_client.stream = stream_side_effect

    from app.providers.yifysubtitles import YifysubtitlesProvider
    provider = YifysubtitlesProvider(mock_client)
    data = await provider.download_archive("https://yifysubtitles.ch/subtitle/x.zip")

    assert data == b"PK\x03\x04retried"
