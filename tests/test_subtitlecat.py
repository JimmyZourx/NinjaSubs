"""Unit tests for the SubtitleCat (subtitlecat.com) provider."""

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from app.providers.subtitlecat import SubtitlecatProvider
from app.utils.language import get_subtitlecat_lang_codes

SEARCH_HTML = """
<html><body>
<table class="sub-table">
<tbody>
<tr>
<td><a href="subs/1634/The.Shawshank.Redemption.1994.1080p.x264.YIFY-eng.html">The.Shawshank.Redemption.1994.1080p.x264.YIFY-eng</a> (translated from English)</td>
<td class="sub-table__stars">&nbsp;</td>
</tr>
<tr>
<td><a href="subs/8/The.Shawshank.Redemption.1994.1080p.x264.YIFY.html">The.Shawshank.Redemption.1994.1080p.x264.YIFY</a></td>
<td class="sub-table__stars">&nbsp;</td>
</tr>
</tbody>
</table>
</body></html>
"""

DETAIL_HTML = """
<html><body>
<div class="all-sub">
  <div class="sub-single">
    <span>Arabic</span>
    <a href="/subs/1657/The.Shawshank.Redemption.1994.1080p.x264.YIFY-eng-ar.srt">Download</a>
  </div>
  <div class="sub-single">
    <span>English</span>
    <a href="/subs/1642/The.Shawshank.Redemption.1994.1080p.x264.YIFY-eng-en.srt">Download</a>
  </div>
</div>
</body></html>
"""


def _response(status_code: int = 200, text: str = "", content: bytes = b"") -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    resp.content = content
    return resp


def _mock_client(search_html: str = SEARCH_HTML, detail_html: str = DETAIL_HTML):
    """Create a mock httpx.AsyncClient that works with bounded streaming downloads."""
    captured = {}

    # Use AsyncMock for get() since it's awaited in production code
    async def get_side_effect(url, params=None, headers=None, timeout=None, **kwargs):
        if "index.php" in url:
            captured["url"] = url
            captured["params"] = params
            return _response(200, text=search_html)
        if url.endswith(".srt"):
            return _response(200, content=b"1\n00:00:01,000 --> 00:00:02,000\nHello\n")
        # Only the first search candidate (id 1634) exposes translations.
        if "/subs/1634/" in url:
            return _response(200, text=detail_html)
        return _response(200, text="<html><body>no translations</body></html>")

    mock_get = AsyncMock(side_effect=get_side_effect)

    # stream() is synchronous, returns async context manager
    async def aiter_bytes_search(chunk_size=None):
        yield search_html.encode("utf-8")

    async def aiter_bytes_detail(chunk_size=None):
        yield detail_html.encode("utf-8")

    async def aiter_bytes_srt(chunk_size=None):
        yield b"1\n00:00:01,000 --> 00:00:02,000\nHello\n"

    async def aiter_bytes_empty(chunk_size=None):
        yield b""

    mock_stream_search = AsyncMock(
        __aenter__=AsyncMock(return_value=MagicMock(
            status_code=200, headers={"content-length": str(len(search_html))},
            encoding="utf-8", aiter_bytes=aiter_bytes_search
        )),
        __aexit__=AsyncMock(return_value=None)
    )
    mock_stream_detail = AsyncMock(
        __aenter__=AsyncMock(return_value=MagicMock(
            status_code=200, headers={"content-length": str(len(detail_html))},
            encoding="utf-8", aiter_bytes=aiter_bytes_detail
        )),
        __aexit__=AsyncMock(return_value=None)
    )
    mock_stream_srt = AsyncMock(
        __aenter__=AsyncMock(return_value=MagicMock(
            status_code=200, headers={"content-length": "40"},
            encoding="utf-8", aiter_bytes=aiter_bytes_srt
        )),
        __aexit__=AsyncMock(return_value=None)
    )
    mock_stream_empty = AsyncMock(
        __aenter__=AsyncMock(return_value=MagicMock(
            status_code=200, headers={"content-length": "0"},
            encoding="utf-8", aiter_bytes=aiter_bytes_empty
        )),
        __aexit__=AsyncMock(return_value=None)
    )

    def stream_side_effect(method, url, params=None, headers=None, timeout=None, **kwargs):
        if "index.php" in url:
            captured["url"] = url
            captured["params"] = params
            return mock_stream_search
        if url.endswith(".srt"):
            return mock_stream_srt
        if "/subs/1634/" in url:
            return mock_stream_detail
        return mock_stream_empty

    mock_stream = MagicMock(side_effect=stream_side_effect)

    mock_client = MagicMock()
    mock_client.get = mock_get
    mock_client.stream = mock_stream
    return mock_client, captured


def test_subtitlecat_lang_codes():
    """Language-code mapping includes ISO-639-1 and legacy variants."""
    assert "ar" in get_subtitlecat_lang_codes("ara")
    assert "en" in get_subtitlecat_lang_codes("eng")
    assert "iw" in get_subtitlecat_lang_codes("heb")
    assert "pt-br" in get_subtitlecat_lang_codes("por")


@pytest.mark.asyncio
async def test_subtitlecat_search_arabic():
    """Arabic search returns the pre-generated -ar.srt translation."""
    mock_client, captured = _mock_client()
    provider = SubtitlecatProvider(mock_client)

    subs = await provider.search_subtitles(
        imdb_id="tt0111161",
        is_series=False,
        title="The Shawshank Redemption",
        year=1994,
        languages=["ara"],
    )

    assert len(subs) == 1
    sub = subs[0]
    assert sub.provider == "subtitlecat"
    assert sub.lang == "ara"
    assert sub.release_name == "The.Shawshank.Redemption.1994.1080p.x264.YIFY-eng"
    assert sub.download_url.endswith(".srt")
    assert sub.download_url.endswith("-eng-ar.srt")
    assert sub.download_url.startswith("https://www.subtitlecat.com/subs/")


@pytest.mark.asyncio
async def test_subtitlecat_search_english():
    """English search picks the -en.srt translation."""
    mock_client, captured = _mock_client()
    provider = SubtitlecatProvider(mock_client)

    subs = await provider.search_subtitles(
        imdb_id="tt0111161",
        is_series=False,
        title="The Shawshank Redemption",
        year=1994,
        languages=["eng"],
    )

    assert len(subs) == 1
    assert subs[0].lang == "eng"
    assert subs[0].download_url.endswith("-eng-en.srt")


@pytest.mark.asyncio
async def test_subtitlecat_no_title_returns_empty():
    """Without a title (e.g. Cinemeta failed) no request is made."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    provider = SubtitlecatProvider(mock_client)

    subs = await provider.search_subtitles(imdb_id="tt0111161", is_series=False, title=None)
    assert subs == []
    mock_client.get.assert_not_called()


@pytest.mark.asyncio
async def test_subtitlecat_no_results():
    """A search page with no result rows yields no releases."""
    mock_client, captured = _mock_client(search_html="<html><body>No results</body></html>")
    provider = SubtitlecatProvider(mock_client)

    subs = await provider.search_subtitles(
        imdb_id="tt0111161", is_series=False, title="Unknown Movie", languages=["ara"]
    )
    assert subs == []


@pytest.mark.asyncio
async def test_subtitlecat_series_query_and_download():
    """Series queries build an SxxExx query and downloads return raw srt bytes."""
    mock_client, captured = _mock_client()
    provider = SubtitlecatProvider(mock_client)

    subs = await provider.search_subtitles(
        imdb_id="tt0903747",
        is_series=True,
        season=5,
        episode=16,
        title="Breaking Bad",
        languages=["ara"],
    )
    assert "Breaking+Bad+S05E16" in captured["url"]
    assert len(subs) == 1

    data = await provider.download_archive(subs[0].download_url)
    assert data is not None
    assert b"Hello" in data
