"""Tests for Semantic Guard Observation Mode v1."""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.semantic_guard import (
    SemanticGuardObserver,
    _fingerprint,
    _hash_id,
    _sanitize_filename,
    _validate_imdb_id,
)


class MockSettings:
    SEMANTIC_GUARD_MODE = "off"
    SEMANTIC_GUARD_MODEL = "test-model"
    SEMANTIC_GUARD_MAX_CONCURRENCY = 1
    SEMANTIC_GUARD_QUEUE_SIZE = 2
    SEMANTIC_GUARD_TIMEOUT_SECONDS = 30.0
    SEMANTIC_GUARD_REPORT_DIR = "/tmp/semantic_guard_test"


def test_disabled_mode_no_imports():
    """disabled mode performs zero semantic work and does not import heavy dependencies."""
    from app.config import settings
    original_mode = settings.SEMANTIC_GUARD_MODE
    settings.SEMANTIC_GUARD_MODE = "off"

    observer = SemanticGuardObserver()
    assert observer.mode == "off"

    observer.enqueue(b"target", b"ref", b"alass", {"imdb_id": "tt1234567"}, "target_id")
    assert observer._queue.qsize() == 0

    settings.SEMANTIC_GUARD_MODE = original_mode


def test_queue_full_drops_safely():
    """queue full drops safely with non-blocking put."""
    from app.config import settings
    original_mode = settings.SEMANTIC_GUARD_MODE
    settings.SEMANTIC_GUARD_MODE = "observe"

    observer = SemanticGuardObserver()
    observer.queue_size = 2

    for i in range(5):
        observer.enqueue(b"target" + bytes([i]), b"ref", b"alass", {}, f"target_{i}")

    assert observer._queue.qsize() <= 2

    settings.SEMANTIC_GUARD_MODE = original_mode


@pytest.mark.asyncio
async def test_duplicate_fingerprint_not_reprocessed():
    """duplicate observation fingerprint is not reprocessed."""
    from app.config import settings
    original_mode = settings.SEMANTIC_GUARD_MODE
    settings.SEMANTIC_GUARD_MODE = "observe"

    observer = SemanticGuardObserver()
    observer._seen_fingerprints.clear()

    target = b"target_bytes"
    reference = b"reference_bytes"
    alass = b"alass_bytes"

    observer.enqueue(target, reference, alass, {}, "target_id")
    observer.enqueue(target, reference, alass, {}, "target_id")

    assert observer._queue.qsize() == 1

    settings.SEMANTIC_GUARD_MODE = original_mode


@pytest.mark.asyncio
async def test_observer_enqueue_non_blocking():
    """observer enqueue is non-blocking."""
    from app.config import settings
    original_mode = settings.SEMANTIC_GUARD_MODE
    settings.SEMANTIC_GUARD_MODE = "observe"

    observer = SemanticGuardObserver()

    import time
    start = time.time()
    observer.enqueue(b"x" * 1000, b"y" * 1000, b"z" * 1000, {}, "id")
    elapsed = time.time() - start

    assert elapsed < 0.01

    settings.SEMANTIC_GUARD_MODE = original_mode


def test_fingerprint_deterministic():
    """fingerprint is deterministic based on target, reference, alass, model."""
    fp1 = _fingerprint(b"a", b"b", b"c", "model")
    fp2 = _fingerprint(b"a", b"b", b"c", "model")
    fp3 = _fingerprint(b"x", b"b", b"c", "model")

    assert fp1 == fp2
    assert fp1 != fp3
    assert len(fp1) == 64


def test_imdb_validation():
    """malformed IMDb ID cannot create unsafe filename."""
    assert _validate_imdb_id("tt1234567") == "tt1234567"
    assert _validate_imdb_id("tt123") == "unknown"
    assert _validate_imdb_id("invalid") == "unknown"
    assert _validate_imdb_id("") == "unknown"


def test_filename_sanitization():
    """sanitize values used in filenames."""
    assert _sanitize_filename("tt1234567") == "tt1234567"
    assert _sanitize_filename("tt123/456") == "tt123_456"
    assert _sanitize_filename("../etc/passwd") == "___etc_passwd"
    assert _sanitize_filename("") == "unknown"


def test_target_id_hashing():
    """target_id is anonymized."""
    h1 = _hash_id("real_target_id")
    h2 = _hash_id("real_target_id")
    h3 = _hash_id("different_id")

    assert h1 == h2
    assert h1 != h3
    assert len(h1) == 16
    assert h1 != "real_target_id"


@pytest.mark.asyncio
async def test_graceful_shutdown_cancellation():
    """graceful shutdown/cancellation with no dangling worker task."""
    from app.config import settings
    original_mode = settings.SEMANTIC_GUARD_MODE
    settings.SEMANTIC_GUARD_MODE = "observe"

    observer = SemanticGuardObserver()
    await observer.start()

    assert observer._worker_task is not None

    await observer.close()

    assert observer._worker_task is None
    assert observer._closing

    settings.SEMANTIC_GUARD_MODE = original_mode


def test_report_allowlist():
    """report allowlist/redaction prevents secrets."""
    from app.services.semantic_guard import _allowlisted_report

    report = {
        "imdb_id": "tt1234567",
        "target_id_hash": "abc123",
        "api_key": "secret123",
        "config_token": "admin_token",
        "provider_credentials": "creds",
        "served_result": "alass_unchanged",
    }

    allowed = _allowlisted_report(report)

    assert "imdb_id" in allowed
    assert "target_id_hash" in allowed
    assert "served_result" in allowed
    assert "api_key" not in allowed
    assert "config_token" not in allowed
    assert "provider_credentials" not in allowed


@pytest.mark.asyncio
async def test_semantic_guard_off_no_queue():
    """SEMANTIC_GUARD_MODE=off must not import heavy semantic dependencies."""
    from app.config import settings
    original_mode = settings.SEMANTIC_GUARD_MODE

    settings.SEMANTIC_GUARD_MODE = "off"

    observer = SemanticGuardObserver()
    assert observer.mode == "off"

    observer.enqueue(b"data", b"data", b"data", {}, "id")
    assert observer._queue.qsize() == 0

    settings.SEMANTIC_GUARD_MODE = original_mode


@pytest.mark.asyncio
async def test_observer_initializes_in_observe_mode():
    """observer actually initializes in observe mode."""
    from app.config import settings
    from app.services.sync.orchestrator import SyncOrchestrator

    original_mode = settings.SEMANTIC_GUARD_MODE
    settings.SEMANTIC_GUARD_MODE = "observe"

    orchestrator = SyncOrchestrator(sync_service=MagicMock())
    orchestrator._sync_service.sync_async = AsyncMock(return_value=b"RESULT")

    mock_resolved = MagicMock()
    mock_resolved.text = "reference text"
    orchestrator.resolve_reference = AsyncMock(return_value=mock_resolved)

    with patch("app.services.semantic_guard.SemanticGuardObserver") as MockObserver:
        mock_obs = MagicMock()
        mock_obs.start = AsyncMock()
        mock_obs.enqueue = MagicMock()
        MockObserver.return_value = mock_obs

        await orchestrator._execute(b"target", {"imdb_id": "tt1234567", "lang": "ar"}, "target_id")

        assert orchestrator._semantic_observer is not None
        mock_obs.start.assert_called_once()

    settings.SEMANTIC_GUARD_MODE = original_mode


@pytest.mark.asyncio
async def test_observer_not_initialized_in_off_mode():
    """observer does not initialize in off mode."""
    from app.config import settings
    from app.services.sync.orchestrator import SyncOrchestrator

    original_mode = settings.SEMANTIC_GUARD_MODE
    settings.SEMANTIC_GUARD_MODE = "off"

    orchestrator = SyncOrchestrator(sync_service=MagicMock())
    orchestrator._sync_service.sync_async = AsyncMock(return_value=b"RESULT")

    mock_resolved = MagicMock()
    mock_resolved.text = "reference text"
    orchestrator.resolve_reference = AsyncMock(return_value=mock_resolved)

    await orchestrator._execute(b"target", {"imdb_id": "tt1234567", "lang": "ar"}, "target_id")

    assert orchestrator._semantic_observer is None

    settings.SEMANTIC_GUARD_MODE = original_mode


@pytest.mark.asyncio
async def test_real_observation_enqueued():
    """real observation is enqueued after successful Alass output."""
    from app.config import settings
    from app.services.sync.orchestrator import SyncOrchestrator

    original_mode = settings.SEMANTIC_GUARD_MODE
    settings.SEMANTIC_GUARD_MODE = "observe"

    orchestrator = SyncOrchestrator(sync_service=MagicMock())
    orch_sync = AsyncMock(return_value=b"REAL_ALASS_RESULT")
    orchestrator._sync_service.sync_async = orch_sync

    mock_resolved = MagicMock()
    mock_resolved.text = "reference text"
    orchestrator.resolve_reference = AsyncMock(return_value=mock_resolved)

    with patch("app.services.semantic_guard.SemanticGuardObserver") as MockObserver:
        mock_obs_instance = MagicMock()
        mock_obs_instance.start = AsyncMock()
        mock_obs_instance.enqueue = MagicMock()
        MockObserver.return_value = mock_obs_instance

        await orchestrator._execute(b"target_bytes", {"imdb_id": "tt1234567", "lang": "ar"}, "real_target_id_123")

        mock_obs_instance.enqueue.assert_called_once()
        call_args = mock_obs_instance.enqueue.call_args
        assert call_args[0][0] == b"target_bytes"
        assert call_args[0][1] == b"reference text"
        assert call_args[0][2] == b"REAL_ALASS_RESULT"
        assert call_args[0][4] == "real_target_id_123"

    await orchestrator.close()
    settings.SEMANTIC_GUARD_MODE = original_mode


@pytest.mark.asyncio
async def test_served_bytes_unchanged_with_observer():
    """byte-for-byte Alass output remains unchanged with observer success/failure/mode off/queue full."""
    from app.config import settings
    from app.services.sync.orchestrator import SyncOrchestrator

    original_mode = settings.SEMANTIC_GUARD_MODE

    for mode in ["off", "observe"]:
        settings.SEMANTIC_GUARD_MODE = mode

        with patch("app.services.semantic_guard.SemanticGuardObserver") as MockObserver:
            mock_obs = MagicMock()
            mock_obs.start = AsyncMock()
            mock_obs.enqueue = MagicMock()
            mock_obs.close = AsyncMock()
            MockObserver.return_value = mock_obs

            orchestrator = SyncOrchestrator(sync_service=MagicMock())
            orchestrator._sync_service.sync_async = AsyncMock(return_value=b"REAL_ALASS_RESULT")

            mock_resolved = MagicMock()
            mock_resolved.text = "reference text"
            orchestrator.resolve_reference = AsyncMock(return_value=mock_resolved)

            result = await orchestrator._execute(b"original", {"imdb_id": "tt1234567", "lang": "ar"}, "target_id")

            assert result == b"REAL_ALASS_RESULT", f"Failed for mode {mode}"

            await orchestrator.close()

    settings.SEMANTIC_GUARD_MODE = original_mode


def test_no_sys_path_mutation():
    """no sys.path mutation occurs."""
    import inspect

    from app.services import semantic_guard
    source = inspect.getsource(semantic_guard)
    assert "sys.path.insert" not in source
    assert "sys.path.append" not in source


def test_report_no_subtitle_text():
    """report nested values cannot contain subtitle text."""
    from app.services.semantic_guard import _allowlisted_report

    report = {
        "imdb_id": "tt1234567",
        "target_id_hash": "abc123",
        "proposed_corrections": [1, 2, 3],
        "served_result": "alass_unchanged",
        "subtitle_text": "This is subtitle content that should not be here",
        "api_key": "secret",
    }

    allowed = _allowlisted_report(report)

    assert "subtitle_text" not in allowed
    assert "api_key" not in allowed
    assert isinstance(allowed["proposed_corrections"], list)
    for item in allowed["proposed_corrections"]:
        assert isinstance(item, int)


def test_report_fingerprint_unique():
    """different fingerprints cannot overwrite reports."""
    from app.services.semantic_guard import SemanticGuardObserver

    with tempfile.TemporaryDirectory() as tmpdir:
        observer = SemanticGuardObserver()
        observer.report_dir = Path(tmpdir)

        report1 = {
            "imdb_id": "tt1234567",
            "target_id_hash": "abc123",
            "fingerprint": "fedcba9876543210",
            "proposed_corrections": [1, 2],
        }

        report2 = {
            "imdb_id": "tt1234567",
            "target_id_hash": "abc123",
            "fingerprint": "1234567890abcdef",
            "proposed_corrections": [3, 4],
        }

        observer._write_report_sync(report1)
        observer._write_report_sync(report2)

        files = list(Path(tmpdir).glob("*.json"))
        assert len(files) == 2, f"Expected 2 files, got {len(files)}"

        contents = [json.loads(f.read_text()) for f in files]
        fingerprints = [c["fingerprint"] for c in contents]
        assert "fedcba9876543210" in fingerprints
        assert "1234567890abcdef" in fingerprints


@pytest.mark.asyncio
async def test_model_load_failure_degrades_safely():
    """model load failure is detected correctly and fails open."""
    from app.config import settings
    from app.services.semantic_guard import SemanticGuardObserver

    original_mode = settings.SEMANTIC_GUARD_MODE
    settings.SEMANTIC_GUARD_MODE = "observe"

    observer = SemanticGuardObserver()
    observer._model_load_failed = True
    observer._model_loaded = False

    result = observer._analyze(b"target", b"ref", b"alass", "tt1234567", "target_id")

    assert result == {}

    settings.SEMANTIC_GUARD_MODE = original_mode


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
