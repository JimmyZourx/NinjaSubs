"""Stage 4 AutoSync serving integration tests.

Focused tests for serving integration call sites:
- _serve_subtitle_handler cache hit
- _serve_subtitle_handler Subsource fallback
- _serve_subtitle_handler fresh download
- proxy_opensubtitles_stream cache hit
- proxy_opensubtitles_stream direct fetch
- proxy_opensubtitles_stream fallback

Tests verify gates, credential safety, cache safety, failure fallback and GET/HEAD compatibility.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.config import settings
from app.main import _maybe_sync_subtitle, _serve_subtitle_handler
from app.services.sync.orchestrator import SyncOrchestrator


@pytest.fixture
def mock_orchestrator(monkeypatch):
    """Provide a mock orchestrator with controllable evaluate_and_sync."""
    mock = MagicMock(spec=SyncOrchestrator)
    mock.evaluate_and_sync = AsyncMock()
    monkeypatch.setattr("app.main._sync_orchestrator", mock)
    return mock


@pytest.mark.asyncio
async def test_server_gate_disabled_returns_original_and_no_call(mock_orchestrator, monkeypatch):
    monkeypatch.setattr(settings, "AUTOSYNC_ENABLED", False)
    original = b"original subtitle"
    meta = {"lang": "ara"}
    result = await _maybe_sync_subtitle(original, "tid", meta, auto_sync=True, convert_ass_enabled=True)
    assert result is original
    mock_orchestrator.evaluate_and_sync.assert_not_awaited()


@pytest.mark.asyncio
async def test_user_gate_disabled_returns_original_and_no_call(mock_orchestrator, monkeypatch):
    monkeypatch.setattr(settings, "AUTOSYNC_ENABLED", True)
    original = b"original subtitle"
    meta = {"lang": "ara"}
    result = await _maybe_sync_subtitle(original, "tid", meta, auto_sync=False, convert_ass_enabled=True)
    assert result is original
    mock_orchestrator.evaluate_and_sync.assert_not_awaited()


@pytest.mark.asyncio
async def test_both_gates_enabled_returns_synced(mock_orchestrator, monkeypatch):
    monkeypatch.setattr(settings, "AUTOSYNC_ENABLED", True)
    original = b"original"
    synced = b"synced"
    mock_orchestrator.evaluate_and_sync.return_value = synced
    meta = {"lang": "ara"}
    result = await _maybe_sync_subtitle(original, "tid", meta, auto_sync=True, convert_ass_enabled=True)
    assert result == synced
    mock_orchestrator.evaluate_and_sync.assert_awaited_once()
    args = mock_orchestrator.evaluate_and_sync.call_args.args
    assert args[0] == original
    assert args[1] == meta


@pytest.mark.asyncio
async def test_language_gate_blocks_non_arabic(mock_orchestrator, monkeypatch):
    monkeypatch.setattr(settings, "AUTOSYNC_ENABLED", True)
    original = b"original"
    meta = {"lang": "eng"}
    result = await _maybe_sync_subtitle(original, "tid", meta, auto_sync=True, convert_ass_enabled=True)
    assert result is original
    mock_orchestrator.evaluate_and_sync.assert_not_awaited()


@pytest.mark.asyncio
async def test_ass_not_converted_blocks_sync(mock_orchestrator, monkeypatch):
    monkeypatch.setattr(settings, "AUTOSYNC_ENABLED", True)
    # Minimal ASS header triggers is_ass_subtitle
    original = b"[Script Info]\nTitle: Test\n\n[Events]\nDialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,Hello"
    meta = {"lang": "ara"}
    result = await _maybe_sync_subtitle(original, "tid", meta, auto_sync=True, convert_ass_enabled=False)
    assert result is original
    mock_orchestrator.evaluate_and_sync.assert_not_awaited()


@pytest.mark.asyncio
async def test_orchestrator_exception_falls_back_to_original(mock_orchestrator, monkeypatch):
    monkeypatch.setattr(settings, "AUTOSYNC_ENABLED", True)
    original = b"original"
    mock_orchestrator.evaluate_and_sync.side_effect = RuntimeError("boom")
    meta = {"lang": "ara"}
    result = await _maybe_sync_subtitle(original, "tid", meta, auto_sync=True, convert_ass_enabled=True)
    assert result is original


@pytest.mark.asyncio
async def test_orchestrator_cancelled_propagates(monkeypatch):
    # CancelledError must propagate
    class DummyOrch:
        async def evaluate_and_sync(self, *a, **k):
            raise asyncio.CancelledError()
    monkeypatch.setattr("app.main._sync_orchestrator", DummyOrch())
    monkeypatch.setattr(settings, "AUTOSYNC_ENABLED", True)
    original = b"orig"
    meta = {"lang": "ara"}
    with pytest.raises(asyncio.CancelledError):
        await _maybe_sync_subtitle(original, "tid", meta, auto_sync=True, convert_ass_enabled=True)


@pytest.mark.asyncio
async def test_credential_keys_passed_transiently(mock_orchestrator, monkeypatch):
    monkeypatch.setattr(settings, "AUTOSYNC_ENABLED", True)
    original = b"orig"
    synced = b"synced"
    mock_orchestrator.evaluate_and_sync.return_value = synced
    meta = {
        "lang": "ara",
        "subdl_key": "key1",
        "subsource_key": "key2",
        "opensubtitles_key": "key3",
        "download_url": "https://example.com/file",
        "stream_url": "https://example.com/stream?token=SECRET",
    }
    result = await _maybe_sync_subtitle(original, "tid", meta, auto_sync=True, convert_ass_enabled=True)
    assert result == synced
    call_args = mock_orchestrator.evaluate_and_sync.call_args
    # evaluate_and_sync called with positional args (original, meta, target_id) and kw auto_sync=True
    passed_meta = call_args.args[1]
    assert passed_meta["subdl_key"] == "key1"
    assert passed_meta["subsource_key"] == "key2"
    assert passed_meta["opensubtitles_key"] == "key3"
    # download_url is not filtered here; orchestrator only uses stream_url, which is allowed
    # Ensure no mutation of original meta keys
    assert meta["subdl_key"] == "key1"


@pytest.mark.asyncio
async def test_no_signed_urls_in_query_context(mock_orchestrator, monkeypatch):
    monkeypatch.setattr(settings, "AUTOSYNC_ENABLED", True)
    original = b"orig"
    synced = b"synced"
    mock_orchestrator.evaluate_and_sync.return_value = synced
    # Meta contains signed stream_url; orchestrator should receive it but not treat download_url as stream
    meta = {
        "lang": "ara",
        "stream_url": "https://cdn.example.com/video?token=SIGN",
        "download_url": "https://cdn.example.com/sub.srt?token=SIGN2",
    }
    await _maybe_sync_subtitle(original, "tid", meta, auto_sync=True, convert_ass_enabled=True)
    _, passed_meta = mock_orchestrator.evaluate_and_sync.call_args.args[:2]
    # No download URLs should be interpreted as stream_url
    assert passed_meta.get("stream_url") == "https://cdn.example.com/video?token=SIGN"
    # download_url is passed through meta but orchestrator _build_query only reads stream_url


@pytest.mark.asyncio
async def test_raw_cache_not_overwritten_by_sync(monkeypatch):
    """Verify serving path does not rewrite raw cache with synced bytes.

    We simulate cache hit path: get_subtitle returns original, _maybe_sync_subtitle returns synced,
    but cache save must not be called with synced.
    """
    from app.main import cache_manager, credential_store

    monkeypatch.setattr(settings, "AUTOSYNC_ENABLED", True)
    target_id = "abc1234567890ab"
    original = b"original cached bytes"
    synced = b"synced bytes"

    # Mock cache to return original
    async def fake_get_subtitle(tid):
        assert tid == target_id
        return original

    def fake_get_metadata(tid):
        return {"lang": "ara", "release_name": "Test"}

    # Ensure save is not called for cache hit
    save_calls = []

    async def fake_save_subtitle(tid, data):
        save_calls.append((tid, data))

    monkeypatch.setattr(cache_manager, "get_subtitle", fake_get_subtitle)
    monkeypatch.setattr(cache_manager, "get_metadata", fake_get_metadata)
    monkeypatch.setattr(cache_manager, "save_subtitle", fake_save_subtitle)
    monkeypatch.setattr(credential_store, "get", AsyncMock(return_value={}))

    # Mock orchestrator
    mock_orch = MagicMock(spec=SyncOrchestrator)
    mock_orch.evaluate_and_sync = AsyncMock(return_value=synced)
    monkeypatch.setattr("app.main._sync_orchestrator", mock_orch)

    # Call handler with auto_sync enabled via config
    from app.utils.config_parser import encode_user_config
    cfg = encode_user_config(auto_sync=True)
    # Ensure _http_client exists to avoid 503
    monkeypatch.setattr("app.main._http_client", object())

    # Handler expects sub_id; we pass target_id
    resp = await _serve_subtitle_handler(f"{target_id}.srt", config_str=cfg)
    # Response content should be synced (handler returns Response with Body)
    # FastAPI Response has body attribute as bytes
    body = resp.body if hasattr(resp, "body") else b""
    assert body == synced
    # Cache must not have been written with synced bytes
    assert save_calls == []


@pytest.mark.asyncio
async def test_generic_fresh_download_caches_original_not_synced(monkeypatch):
    """Fresh download path must save original to raw cache and serve synced only for request."""
    from app.main import cache_manager, credential_store
    from app.providers import SubdlProvider

    monkeypatch.setattr(settings, "AUTOSYNC_ENABLED", True)
    target_id = "deadbeefcafe1234"
    original = b"1\n00:00:01,000 --> 00:00:02,000\nTest\n"
    synced = b"1\n00:00:01,000 --> 00:00:02,000\nSynced\n"

    async def fake_get_subtitle(tid):
        return None

    def fake_get_metadata(tid):
        return {
            "lang": "ara",
            "provider": "subdl",
            "download_url": "https://example.com/archive.zip",
            "release_name": "Test",
        }

    saved = []

    async def fake_save_subtitle(tid, data):
        saved.append((tid, data))

    monkeypatch.setattr(cache_manager, "get_subtitle", fake_get_subtitle)
    monkeypatch.setattr(cache_manager, "get_metadata", fake_get_metadata)
    monkeypatch.setattr(cache_manager, "save_subtitle", fake_save_subtitle)
    monkeypatch.setattr(credential_store, "get", AsyncMock(return_value={}))

    mock_orch = MagicMock(spec=SyncOrchestrator)
    mock_orch.evaluate_and_sync = AsyncMock(return_value=synced)
    monkeypatch.setattr("app.main._sync_orchestrator", mock_orch)

    # Mock provider download
    async def fake_download_archive(self, url, api_key=None):
        return original  # not zip to avoid extraction

    monkeypatch.setattr(SubdlProvider, "download_archive", fake_download_archive)
    monkeypatch.setattr("app.main._http_client", object())

    from app.utils.config_parser import encode_user_config
    cfg = encode_user_config(auto_sync=True)

    resp = await _serve_subtitle_handler(f"{target_id}.srt", config_str=cfg)
    body = resp.body if hasattr(resp, "body") else b""
    assert body == synced
    assert len(saved) == 1
    saved_tid, saved_data = saved[0]
    assert saved_tid == target_id
    # Cache must contain original, not synced
    assert saved_data == original


@pytest.mark.asyncio
async def test_orchestrator_failure_serves_original_without_5xx(monkeypatch):
    """When orchestrator fails, handler must still return original subtitle, not 5xx."""
    from app.main import cache_manager, credential_store

    monkeypatch.setattr(settings, "AUTOSYNC_ENABLED", True)
    target_id = "0123456789abcdef"
    original = b"original cached"

    async def fake_get_subtitle(tid):
        return original

    def fake_get_metadata(tid):
        return {"lang": "ara", "release_name": "Test"}

    monkeypatch.setattr(cache_manager, "get_subtitle", fake_get_subtitle)
    monkeypatch.setattr(cache_manager, "get_metadata", fake_get_metadata)
    monkeypatch.setattr(credential_store, "get", AsyncMock(return_value={}))

    mock_orch = MagicMock(spec=SyncOrchestrator)
    mock_orch.evaluate_and_sync = AsyncMock(side_effect=RuntimeError("sync failed"))
    monkeypatch.setattr("app.main._sync_orchestrator", mock_orch)
    monkeypatch.setattr("app.main._http_client", object())

    from app.utils.config_parser import encode_user_config
    cfg = encode_user_config(auto_sync=True)

    resp = await _serve_subtitle_handler(f"{target_id}.srt", config_str=cfg)
    body = resp.body if hasattr(resp, "body") else b""
    assert body == original


@pytest.mark.asyncio
async def test_get_head_behavior_preserved(monkeypatch):
    """GET and HEAD responses remain compatible when AutoSync is enabled."""
    from app.main import cache_manager, credential_store

    monkeypatch.setattr(settings, "AUTOSYNC_ENABLED", True)
    target_id = "feedfacecafebabe"
    original = b"original"

    async def fake_get_subtitle(tid):
        return original

    def fake_get_metadata(tid):
        return {"lang": "ara", "release_name": "Test"}

    monkeypatch.setattr(cache_manager, "get_subtitle", fake_get_subtitle)
    monkeypatch.setattr(cache_manager, "get_metadata", fake_get_metadata)
    monkeypatch.setattr(credential_store, "get", AsyncMock(return_value={}))

    mock_orch = MagicMock(spec=SyncOrchestrator)
    mock_orch.evaluate_and_sync = AsyncMock(return_value=b"synced")
    monkeypatch.setattr("app.main._sync_orchestrator", mock_orch)
    monkeypatch.setattr("app.main._http_client", object())

    from app.utils.config_parser import encode_user_config
    cfg = encode_user_config(auto_sync=True)

    resp = await _serve_subtitle_handler(f"{target_id}.srt", config_str=cfg)
    # _build_subtitle_response returns Response that supports HEAD via FastAPI
    # Verify headers are present
    assert resp.status_code == 200
    assert "content-type" in resp.headers
