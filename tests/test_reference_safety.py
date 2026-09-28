"""Stage 2 HTTP, credential-boundary, hash-confirmation, and cache-safety tests."""

import json
from collections import deque

import pytest

from app.providers.opensubtitles import OpenSubtitlesProvider
from app.providers.subdl import SubdlProvider
from app.providers.subsource import SubsourceProvider
from app.services.sync.cache import ReferenceDiskCache
from app.services.sync.hash_strategy import HashExactStrategy
from app.services.sync.query import ReferenceQuery
from app.utils.http_limits import (
    MAX_HTML_RESPONSE_BYTES,
    bounded_download_bytes,
    bounded_fetch_json,
)


class Response:
    def __init__(self, status=200, chunks=(), headers=None):
        self.status_code = status
        self.chunks = list(chunks)
        self.headers = dict(headers or {})
        self.encoding = "utf-8"
        self.iterated = False
        self.yielded = 0
        self.chunk_sizes = []

    async def aiter_bytes(self, chunk_size=None):
        self.iterated = True
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
    def __init__(self, responses=()):
        self.responses = deque(responses)
        self.calls = []

    def stream(self, method, url, **kwargs):
        self.calls.append((method, str(url), kwargs))
        value = self.responses.popleft()
        if callable(value):
            value = value(method, str(url), kwargs)
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
async def test_bounded_json_rejects_oversized_metadata_without_body_iteration():
    response = Response(200, [b"{}"], {"content-length": str(MAX_HTML_RESPONSE_BYTES + 1)})
    client = Client([response])
    status, payload = await bounded_fetch_json(
        client,
        "GET",
        "https://api.opensubtitles.com/api/v1/subtitles",
        allowed_hosts={"api.opensubtitles.com"},
    )
    assert status == 200
    assert payload is None
    assert response.iterated is False


@pytest.mark.asyncio
async def test_bounded_json_rejects_automatic_redirect_and_private_redirect():
    redirect = Response(302, [], {"location": "https://127.0.0.1/admin"})
    client = Client([redirect])
    status, payload = await bounded_fetch_json(
        client,
        "GET",
        "https://api.opensubtitles.com/api/v1/subtitles",
        headers={"Api-Key": "secret"},
        allowed_hosts={"api.opensubtitles.com"},
        follow_redirects=True,
    )
    assert status == 302
    assert payload is None
    assert len(client.calls) == 1


@pytest.mark.parametrize("private_ip", ["127.0.0.1", "192.168.1.10", "169.254.169.254"])
@pytest.mark.asyncio
async def test_allowlisted_hostname_resolving_private_is_rejected_before_request(
    monkeypatch, private_ip
):
    from app.utils import http_limits

    monkeypatch.setattr(
        http_limits.socket,
        "getaddrinfo",
        lambda host, port, **kwargs: [(2, 1, 6, "", (private_ip, port or 443))],
    )
    client = Client([])
    status, data = await bounded_download_bytes(
        client,
        "https://api.subdl.com/file.zip",
        headers={"x-api-key": "SECRET_API_KEY", "Authorization": "Bearer SECRET_TOKEN"},
        allowed_hosts={"api.subdl.com"},
    )
    assert (status, data) == (0, b"")
    assert client.calls == []
    assert "SECRET_API_KEY" not in repr(client.calls)
    assert "SECRET_TOKEN" not in repr(client.calls)


@pytest.mark.asyncio
async def test_allowlisted_hostname_with_public_dns_reaches_stream(monkeypatch):
    from app.utils import http_limits

    monkeypatch.setattr(
        http_limits.socket,
        "getaddrinfo",
        lambda host, port, **kwargs: [(2, 1, 6, "", ("93.184.216.34", port or 443))],
    )
    payload = b"bounded public response"
    client = Client([Response(200, [payload], {"content-length": str(len(payload))})])
    status, data = await bounded_download_bytes(
        client,
        "https://api.subdl.com/file.zip",
        headers={"x-api-key": "transient-key"},
        allowed_hosts={"api.subdl.com"},
    )
    assert (status, data) == (200, payload)
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_subsource_authenticated_api_search_does_not_follow_redirect():
    redirect = Response(302, [], {"location": "https://files.subsource.net/steal"})
    client = Client([redirect])
    result = await SubsourceProvider(client).search_subtitles(
        "tt1", api_key="transient-secret", languages=["eng"]
    )
    assert result == []
    assert len(client.calls) == 1
    assert client.calls[0][2]["headers"]["X-API-Key"] == "transient-secret"
    assert client.calls[0][2]["follow_redirects"] is False


@pytest.mark.asyncio
async def test_subdl_authenticated_api_search_does_not_follow_redirect():
    redirect = Response(302, [], {"location": "https://dl.subdl.com/steal"})
    client = Client([redirect])
    result = await SubdlProvider(client).search_subtitles(
        "tt1", api_key="transient-secret", languages=["eng"]
    )
    assert result == []
    assert len(client.calls) == 1
    assert client.calls[0][2]["follow_redirects"] is False
    assert client.calls[0][2]["params"]["api_key"] == "transient-secret"


@pytest.mark.asyncio
async def test_authenticated_provider_initial_url_rejects_untrusted_host_before_request():
    client = Client([])
    result = await SubsourceProvider(client).download_archive(
        "https://evil.example/subtitles/12/download", api_key="SECRET_API_KEY"
    )
    assert result is None
    assert client.calls == []


@pytest.mark.asyncio
async def test_subdl_never_sends_credentials_to_untrusted_initial_release_url():
    client = Client([])
    result = await SubdlProvider(client).download_archive(
        "https://evil.example/steal", api_key="SECRET_SUBDL_KEY"
    )
    assert result is None
    assert client.calls == []


@pytest.mark.asyncio
async def test_subsource_authenticated_http_redirect_to_cdn_fails_closed():
    first = Response(302, [], {"location": "https://files.subsource.net/subtitle.srt?api_key=LEAK"})
    client = Client([first])
    result = await SubsourceProvider(client).download_archive("22", api_key="SECRET_API_KEY")
    assert result is None
    assert len(client.calls) == 1
    assert first.iterated is False


@pytest.mark.asyncio
async def test_subsource_cdn_json_download_url_gets_no_api_credentials():
    metadata = json.dumps({"downloadUrl": "https://files.subsource.net/signed.srt?token=SIGNED"}).encode()
    final = b"1\n00:00:01,000 --> 00:00:02,000\nReference\n"
    client = Client([
        Response(200, [metadata], {"content-length": str(len(metadata))}),
        Response(200, [final], {"content-length": str(len(final))}),
    ])
    result = await SubsourceProvider(client).download_archive("77", api_key="SECRET_API_KEY")
    assert result == final
    headers = {key.lower() for key in client.calls[1][2]["headers"]}
    assert "x-api-key" not in headers
    assert "authorization" not in headers
    assert "SECRET_API_KEY" not in repr(client.calls[1][2])


@pytest.mark.asyncio
async def test_subsource_reference_search_uses_bounded_json_api_calls():
    movie_json = json.dumps({"movieId": 123}).encode()
    subtitle_json = json.dumps({
        "data": [{
            "id": "eng-id",
            "release_name": "Movie.2024.1080p.BluRay.x264-GRP",
            "language": "English",
            "download_path": "/subtitles/eng-id/download",
        }]
    }).encode()
    client = Client([
        Response(200, [movie_json], {"content-length": str(len(movie_json))}),
        Response(200, [subtitle_json], {"content-length": str(len(subtitle_json))}),
    ])
    releases = await SubsourceProvider(client).search_subtitles(
        "tt1", api_key="transient-subsource-key", languages=["eng"]
    )
    assert len(releases) == 1
    assert [call[0] for call in client.calls] == ["GET", "GET"]
    assert all(call[2]["follow_redirects"] is False for call in client.calls)
    assert client.calls[0][2]["headers"]["X-API-Key"] == "transient-subsource-key"


@pytest.mark.asyncio
async def test_subsource_search_metadata_body_is_bounded():
    oversized = Response(200, [b"{}"], {"content-length": str(MAX_HTML_RESPONSE_BYTES + 1)})
    client = Client([oversized])
    result = await SubsourceProvider(client).search_subtitles(
        "tt1", api_key="transient", title="Movie", languages=["eng"]
    )
    assert result == []
    assert oversized.iterated is False


@pytest.mark.asyncio
async def test_subdl_search_metadata_body_is_bounded_and_auth_stays_on_api_origin():
    oversized = Response(200, [b"{}"], {"content-length": str(MAX_HTML_RESPONSE_BYTES + 1)})
    client = Client([oversized])
    result = await SubdlProvider(client).search_subtitles(
        "tt1", api_key="transient", languages=["eng"]
    )
    assert result == []
    assert oversized.iterated is False
    assert len(client.calls) == 1
    assert client.calls[0][1] == "https://api.subdl.com/api/v1/subtitles"
    assert client.calls[0][2]["follow_redirects"] is False


@pytest.mark.asyncio
async def test_opensubtitles_hash_strategy_ignores_unconfirmed_candidates(tmp_path):
    body = {
        "data": [
            {"attributes": {
                "language": "en", "release": "Show.1080p.WEB-DL",
                "files": [{"file_id": 2, "file_name": "Show.srt"}],
                "moviehash_match": False,
            }}
        ]
    }
    client = Client([Response(200, [json.dumps(body).encode()])])
    provider = OpenSubtitlesProvider(client)
    strategy = HashExactStrategy(provider, cache=ReferenceDiskCache(tmp_path, min_bytes=10))
    result = await strategy.resolve(
        ReferenceQuery(imdb_id="tt1", video_hash="hash", api_keys={"opensubtitles": "transient"})
    )
    assert result is None
    assert len(client.calls) == 1  # Search only; no /download metadata POST or archive request.
    assert client.calls[0][2]["params"]["moviehash"] == "hash"
    assert client.calls[0][2]["headers"]["Api-Key"] == "transient"


@pytest.mark.asyncio
async def test_opensubtitles_confirmed_hash_uses_bounded_post_and_archive_stream(tmp_path):
    search_data = {"data": [{"attributes": {
        "language": "en", "release": "Show.1080p.WEB-DL",
        "files": [{"file_id": 22, "file_name": "Show.srt"}],
        "moviehash_match": True,
    }}]}
    link_data = {"link": "https://download.opensubtitles.com/signed.srt?token=SIGNED_URL"}
    srt = ("1\n00:00:01,000 --> 00:00:02,000\n" + ("reference line\n" * 100)).encode()
    client = Client([
        Response(200, [json.dumps(search_data).encode()]),
        Response(200, [json.dumps(link_data).encode()]),
        Response(200, [srt], {"content-length": str(len(srt))}),
    ])
    provider = OpenSubtitlesProvider(client)
    strategy = HashExactStrategy(
        provider, min_bytes=100, cache=ReferenceDiskCache(tmp_path, min_bytes=100)
    )
    result = await strategy.resolve(
        ReferenceQuery(imdb_id="tt1", video_hash="moviehash", api_keys={"opensubtitles": "transient"})
    )
    assert result and "reference line" in result
    assert [call[0] for call in client.calls] == ["GET", "POST", "GET"]
    assert client.calls[1][2]["json"] == {"file_id": 22}
    assert client.calls[1][2]["headers"]["Api-Key"] == "transient"
    assert client.calls[2][1].startswith("https://download.opensubtitles.com/")


@pytest.mark.asyncio
async def test_opensubtitles_link_metadata_content_length_is_bounded():
    oversized_metadata = Response(
        200,
        [json.dumps({"link": "https://download.opensubtitles.com/signed.srt?token=NEVER_READ"}).encode()],
        {"content-length": str(MAX_HTML_RESPONSE_BYTES + 1)},
    )
    client = Client([oversized_metadata])
    result = await OpenSubtitlesProvider(client).download_archive(
        "/sub/opensubtitles/22.srt", api_key="transient"
    )
    assert result is None
    assert len(client.calls) == 1
    assert client.calls[0][0] == "POST"
    assert client.calls[0][1].endswith("/api/v1/download")
    assert oversized_metadata.iterated is False


@pytest.mark.asyncio
async def test_opensubtitles_search_metadata_body_is_bounded():
    oversized = Response(200, [b"{}"], {"content-length": str(MAX_HTML_RESPONSE_BYTES + 1)})
    client = Client([oversized])
    results = await OpenSubtitlesProvider(client).search_subtitles(
        "tt1", api_key="transient", languages=["eng"]
    )
    assert results == []
    assert oversized.iterated is False
    assert client.calls[0][2]["follow_redirects"] is False


def test_reference_cache_provenance_rejects_provider_url_and_unknown_provider(tmp_path):
    from app.services.sync.cache import ReferenceDiskCache

    cache = ReferenceDiskCache(tmp_path, min_bytes=10)
    query = ReferenceQuery(imdb_id="tt1", target_filename="Movie.2024.BluRay-GRP.mkv")
    text = "1\n00:00:01,000 --> 00:00:02,000\n" + ("line\n" * 10)
    cache.set(
        query,
        "subdl",
        text,
        kind="team",
        candidate="https://download.example/file.srt?token=SIGNED",
    )
    cached = cache.get(query)
    assert cached is not None
    assert cached.candidate == ""
    assert list(tmp_path.glob("*unknown-provider*")) == []
    cache.set(query, "unknown-provider", text, kind="team")
    assert len(list(tmp_path.glob("*.srt"))) == 1


@pytest.mark.parametrize(
    "candidate",
    [
        "https://example.test/file",
        "http://example.test/file",
        "//example.test/file",
        "ftp://example.test/file",
        "?token=secret",
        "?sig=secret",
        "?signature=secret",
        "?api_key=secret",
        "?api-key=secret",
        "?apikey=secret",
        "?x-amz-signature=secret",
    ],
)
def test_reference_cache_never_persists_url_or_signed_candidate(tmp_path, candidate):
    cache = ReferenceDiskCache(tmp_path, min_bytes=10)
    query = ReferenceQuery(imdb_id="tt1", target_filename="Movie.2024.BluRay-GRP.mkv")
    text = "1\n00:00:01,000 --> 00:00:02,000\n" + ("line\n" * 10)
    cache.set(query, "subdl", text, kind="team", candidate=candidate)
    sidecar = json.loads(next(tmp_path.glob("*.srt.json")).read_text(encoding="utf-8"))
    assert sidecar["candidate"] == ""


@pytest.mark.parametrize(
    "candidate",
    ["Show.S01E02.1080p.BluRay.x264-GRP", "Movie: Directors Cut 2160p REMUX"],
)
def test_reference_cache_keeps_normal_release_name_provenance(tmp_path, candidate):
    cache = ReferenceDiskCache(tmp_path, min_bytes=10)
    query = ReferenceQuery(imdb_id="tt1", target_filename="Movie.2024.BluRay-GRP.mkv")
    text = "1\n00:00:01,000 --> 00:00:02,000\n" + ("line\n" * 10)
    cache.set(query, "subdl", text, kind="team", candidate=candidate)
    assert cache.get(query).candidate == candidate


@pytest.mark.asyncio
async def test_signed_opensubtitles_link_never_appears_in_logs(caplog):
    link = "https://download.opensubtitles.com/file.srt?token=DO_NOT_LOG"
    payload = b"1\n00:00:01,000 --> 00:00:02,000\ntext\n"
    client = Client([
        Response(200, [json.dumps({"link": link}).encode()]),
        Response(200, [payload], {"content-length": str(len(payload))}),
    ])
    with caplog.at_level("INFO"):
        await OpenSubtitlesProvider(client).download_archive(
            "/sub/opensubtitles/22.srt", api_key="transient"
        )
    assert "DO_NOT_LOG" not in caplog.text
    assert link not in caplog.text
