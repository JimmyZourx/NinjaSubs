"""Stage 3 preserves the foundation gates and byte-for-byte fallback."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from app.config import settings
from app.extractor import MAX_SUBTITLE_ENTRY_BYTES
from app.services.sync.orchestrator import SyncOrchestrator
from app.services.sync.query import ResolvedReference

pytestmark = pytest.mark.asyncio

ORIGINAL = b"original \xff bytes"
SYNCED = "1\n00:00:01,000 --> 00:00:02,000\nمرحبا\n".encode()


@pytest.fixture
def setup(monkeypatch):
    monkeypatch.setattr(settings, "AUTOSYNC_ENABLED", True)
    strategy = type("Strategy", (), {})()
    strategy.resolve_with_provenance = AsyncMock(return_value=ResolvedReference("English reference"))
    service = type("Service", (), {})()
    service.sync_async = AsyncMock(return_value=SYNCED)
    orch = SyncOrchestrator(external_strategy=strategy, sync_service=service)
    return orch, service, strategy


async def test_success_and_release_never_passed_to_runner(setup):
    orch, service, _ = setup
    release = "../../$(touch sentinel); & |"
    assert await orch.evaluate_and_sync(ORIGINAL, {"lang": "ara", "release_name": release}, "id", True) == SYNCED
    args = service.sync_async.call_args
    assert len(args.args) == 2
    assert all(isinstance(arg, bytes) for arg in args.args)
    assert release.encode() not in args.args
    await orch.close()


@pytest.mark.parametrize("failure", [None, b"", FileNotFoundError(), OSError(), ValueError()])
async def test_exact_original_on_failure(setup, failure):
    orch, service, _ = setup
    if isinstance(failure, Exception):
        service.sync_async.side_effect = failure
    else:
        service.sync_async.return_value = failure
    assert await orch.evaluate_and_sync(ORIGINAL, {"lang": "ara"}, "id", True) is ORIGINAL
    await orch.close()


@pytest.mark.parametrize("gate", ["server", "user", "language", "service", "reference", "oversize"])
async def test_gates_preserved(setup, monkeypatch, gate):
    orch, service, strategy = setup
    auto, language, data = True, "ara", ORIGINAL
    if gate == "server":
        monkeypatch.setattr(settings, "AUTOSYNC_ENABLED", False)
    elif gate == "user":
        auto = False
    elif gate == "language":
        language = "eng"
    elif gate == "service":
        orch._sync_service = None
    elif gate == "reference":
        strategy.resolve_with_provenance.return_value = ResolvedReference(None)
    else:
        data = b"x" * (MAX_SUBTITLE_ENTRY_BYTES + 1)
    assert await orch.evaluate_and_sync(data, {"lang": language}, "id", auto) is data
    service.sync_async.assert_not_awaited()
    await orch.close()


async def test_singleflight_preserved(setup):
    orch, service, _ = setup
    started, finish = asyncio.Event(), asyncio.Event()
    async def sync(*args):
        started.set()
        await finish.wait()
        return SYNCED
    service.sync_async.side_effect = sync
    tasks = [asyncio.create_task(orch.evaluate_and_sync(ORIGINAL, {"lang": "ara"}, "id", True)) for _ in range(2)]
    await started.wait()
    await asyncio.sleep(0)
    finish.set()
    assert await asyncio.gather(*tasks) == [SYNCED, SYNCED]
    service.sync_async.assert_awaited_once()
    await orch.close()


async def test_cancellation_not_swallowed(setup):
    orch, service, _ = setup
    service.sync_async.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await orch.evaluate_and_sync(ORIGINAL, {"lang": "ara"}, "id", True)
    await orch.close()
