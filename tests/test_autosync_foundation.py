"""Foundation-only AutoSync query, matching, cache, and gate regressions."""

from types import SimpleNamespace

import pytest

from app.models import SubtitleRelease, UserPreferences
from app.services.sync.cache import ReferenceDiskCache
from app.services.sync.hash_strategy import HashExactStrategy
from app.services.sync.matching import (
    _edition_tags,
    _member_episode_number,
    _release_group,
    _season_number,
    _source_kind,
    is_informative_release_name,
)
from app.services.sync.orchestrator import SyncOrchestrator, build_synced_cache_key
from app.services.sync.query import ReferenceQuery, ResolvedReference
from app.services.sync.tree import decide
from app.utils.config_parser import encode_user_config, parse_user_config


def test_auto_sync_preference_round_trips_base64_json_and_defaults_off():
    assert UserPreferences().auto_sync is False
    assert parse_user_config(encode_user_config(subdl_key="key")).auto_sync is False
    enabled = parse_user_config(encode_user_config(subdl_key="key", auto_sync=True))
    assert enabled.auto_sync is True
    disabled = parse_user_config(encode_user_config(auto_sync=False))
    assert disabled.auto_sync is False


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ("auto_sync=true", True),
        ("auto_sync=1", True),
        ("auto_sync=false", False),
        ("auto_sync=off", False),
        ("subdl_key=unchanged&auto_sync=yes", True),
    ],
)
def test_auto_sync_optional_query_config(config, expected):
    prefs = parse_user_config(config)
    assert prefs.auto_sync is expected
    if "subdl_key=unchanged" in config:
        assert prefs.subdl_key == "unchanged"


def test_auto_sync_base64_json_alias_and_false_values():
    import base64
    import json

    def payload(value):
        return base64.urlsafe_b64encode(json.dumps(value).encode()).decode().rstrip("=")

    assert parse_user_config(payload({"autoSync": True})).auto_sync is True
    assert parse_user_config(payload({"auto_sync": "false"})).auto_sync is False


def test_reference_query_series_identity_and_no_credential_fingerprinting():
    first = ReferenceQuery(
        imdb_id="tt123", media_type="series", season=2, episode=3,
        target_filename="Show.S02E03.1080p.BluRay-GRP.mkv", video_hash="HASH",
        api_keys={"subdl": "first-secret"},
    )
    second = ReferenceQuery(
        imdb_id="tt123", media_type="series", season=2, episode=3,
        target_filename="Show.S02E03.1080p.BluRay-GRP.mkv", video_hash="HASH",
        api_keys={"subdl": "other-secret"},
    )
    assert first.is_series is True
    assert first.cache_stem == second.cache_stem
    assert "first-secret" not in first.cache_stem
    assert "other-secret" not in second.cache_stem
    assert first.effective_group == "GRP"


def test_reference_query_hides_transient_secrets_from_repr_and_cache_identity():
    first = ReferenceQuery(
        imdb_id="tt1",
        stream_url="https://example.test/video?token=SECRET_STREAM",
        api_keys={"opensubtitles": "SECRET_API_KEY"},
    )
    second = ReferenceQuery(
        imdb_id="tt1",
        stream_url="https://example.test/video?token=ROTATED_STREAM",
        api_keys={"opensubtitles": "ROTATED_API_KEY"},
    )
    rendered = repr(first)
    assert "SECRET_STREAM" not in rendered
    assert "SECRET_API_KEY" not in rendered
    assert first.cache_stem == second.cache_stem


def test_matching_extracts_source_edition_group_and_episode_signals():
    name = "Show.S02E03.1080p.BluRay.x264-GRP.srt"
    assert _season_number(name) == 2
    assert _member_episode_number(name) == 3
    assert _source_kind(name) == "bluray"
    assert _release_group(name) == "GRP"
    assert _edition_tags("Movie.Extended.1080p.BluRay") == frozenset({"extended"})
    assert is_informative_release_name(name)
    assert not is_informative_release_name("opaque123.mkv")


def test_tree_prefers_confirmed_team_and_aborts_source_or_cut_conflicts():
    query = ReferenceQuery(
        imdb_id="tt1", media_type="series", season=1, episode=2,
        target_filename="Show.S01E02.1080p.BluRay.x264-GRP.mkv",
    )
    team = SimpleNamespace(release_name="Show.S01E02.720p.BluRay.x264-GRP.srt")
    other_team = SimpleNamespace(release_name="Show.S01E02.1080p.WEB-DL.x264-OTHER.srt")
    assert decide([other_team, team], query).kind == "team"
    assert decide(
        [SimpleNamespace(release_name="Show.S01E02.1080p.WEB-DL.x264-GRP.srt")], query
    ).kind == "abort"

    cut_query = ReferenceQuery(
        imdb_id="tt2", target_filename="Movie.2020.Extended.1080p.BluRay.x264-GRP.mkv"
    )
    theatrical = SimpleNamespace(
        release_name="Movie.2020.Theatrical.1080p.BluRay.x264-GRP.srt"
    )
    assert decide([theatrical], cut_query).kind == "abort"


def test_tree_uninformative_target_is_strict_by_default():
    query = ReferenceQuery(imdb_id="tt1", target_filename="opaque.mkv")
    release = SimpleNamespace(release_name="Movie.2024.1080p.BluRay-GRP.srt")
    assert decide([release], query).kind == "abort"
    assert decide([release], query, strict=False).kind == "edition"


def test_reference_disk_cache_roundtrip_and_payload_provenance(tmp_path):
    cache = ReferenceDiskCache(tmp_path, ttl=3600, min_bytes=10)
    query = ReferenceQuery(imdb_id="tt1", target_filename="Movie.2024.1080p.BluRay-GRP.mkv")
    text = "1\n00:00:01,000 --> 00:00:02,000\n" + ("reference text\n" * 3)
    cache.set(query, "opensubtitles", text, kind="hash", candidate="Movie-GRP.srt")
    assert cache.get(query) == ResolvedReference(text, "hash", False, "Movie-GRP.srt")

    # A changed payload cannot be paired with the old verdict sidecar.
    path = next(tmp_path.glob("*.srt"))
    path.write_text(text + "changed\n", encoding="utf-8")
    assert cache.get(query) is None


@pytest.mark.asyncio
async def test_hash_strategy_requires_confirmed_english_hash_match(tmp_path):
    reference = "1\n00:00:01,000 --> 00:00:02,000\n" + ("English reference\n" * 30)

    class FakeProvider:
        def __init__(self, release):
            self.release = release
            self.search_calls = []
            self.downloads = []

        async def search_subtitles(self, **kwargs):
            self.search_calls.append(kwargs)
            return [self.release]

        async def download_archive(self, download_ref, api_key=None):
            self.downloads.append(download_ref)
            return reference.encode()

    query = ReferenceQuery(imdb_id="tt1", video_hash="moviehash", api_keys={"opensubtitles": "key"})
    unconfirmed = SubtitleRelease(
        release_name="Movie.en.srt", download_url="/file/1", provider="opensubtitles",
        lang="eng", is_hash_match=False,
    )
    provider = FakeProvider(unconfirmed)
    strategy = HashExactStrategy(
        provider, min_bytes=10, cache=ReferenceDiskCache(tmp_path, min_bytes=10)
    )
    assert await strategy.resolve(query) is None
    assert provider.downloads == []

    confirmed = unconfirmed.model_copy(update={"is_hash_match": True})
    provider.release = confirmed
    text = await strategy.resolve(query)
    assert text and "English reference" in text
    assert provider.search_calls[-1]["languages"] == ["eng"]
    assert provider.search_calls[-1]["video_hash"] == "moviehash"


@pytest.mark.asyncio
async def test_orchestrator_server_and_user_gates_return_original(monkeypatch):
    from app.config import settings

    payload = b"original subtitle payload"
    strategy = SimpleNamespace(resolve_with_provenance=None)

    async def should_not_run(query):
        raise AssertionError("A disabled AutoSync gate must not resolve references")

    strategy.resolve_with_provenance = should_not_run
    orchestrator = SyncOrchestrator(hash_strategy=strategy)
    monkeypatch.setattr(settings, "AUTOSYNC_ENABLED", False)
    assert await orchestrator.evaluate_and_sync(payload, {"lang": "ara"}, "sub", True) == payload

    monkeypatch.setattr(settings, "AUTOSYNC_ENABLED", True)
    assert await orchestrator.evaluate_and_sync(payload, {"lang": "ara"}, "sub", False) == payload
    assert await orchestrator.evaluate_and_sync(payload, {"lang": "eng"}, "sub", True) == payload


def test_sync_cache_key_fingerprints_fallback_identity_without_secret_inputs():
    meta = {
        "imdb_id": "tt1", "season": 1, "episode": 2, "video_hash": "HASH",
        "target_filename": "Show.S01E02.1080p.BluRay-GROUP.mkv",
        "stream_url": "https://media.test/video?token=stream-secret",
        "subdl_key": "secret",
    }
    a = build_synced_cache_key(meta, "sub", "payload-a", "team")
    b = build_synced_cache_key(meta, "sub", "payload-b", "team")
    c = build_synced_cache_key(meta, "sub", "payload-a", "edition")
    assert a.startswith("final_sub:tt1:s1e2:")
    assert len({a, b, c}) == 3
    assert "secret" not in a
    assert "Show.S01E02" not in a
    assert "stream-secret" not in a
    fingerprint = a.split(":")[3]
    assert len(fingerprint) == 20
    assert all(char in "0123456789abcdef" for char in fingerprint)

    same_identity = {
        **meta,
        "stream_url": "https://media.test/video?token=rotated",
        "subdl_key": "another-key",
    }
    assert build_synced_cache_key(same_identity, "sub", "payload-a", "team") == a

    without_hash = {**meta, "video_hash": "", "stream_url": None}
    fallback_a = build_synced_cache_key(without_hash, "sub", "payload-a", "team")
    fallback_b = build_synced_cache_key(
        {**without_hash, "target_filename": "Different.Release.720p.WEB-DL.mkv"},
        "sub", "payload-a", "team",
    )
    assert fallback_a != fallback_b
    assert "Show.S01E02" not in fallback_a


def test_configure_page_renders_auto_sync_toggle_and_serialization(client):
    body = client.get("/configure").text
    assert 'id="autoSync"' in body
    assert 'id="autoSync" checked' not in body
    assert "configObj.auto_sync = true" in body
    assert "autoSyncEnabled" in body

    enabled_config = encode_user_config(auto_sync=True)
    configured = client.get(f"/{enabled_config}/configure").text
    assert '"auto_sync": true' in configured
    assert "typeof initialPrefs.auto_sync === \"boolean\"" in configured


def test_auto_sync_ui_json_keeps_safe_json_injection_path(client):
    body = client.get("/configure").text
    assert "const initialPrefs =" in body
    # The template still receives phase-two JSON through the existing safe serializer.
    from app.main import _safe_json

    assert "\\u003c" in _safe_json({"value": "</script>"}).lower()
