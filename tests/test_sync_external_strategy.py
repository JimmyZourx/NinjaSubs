"""Reference lookup ordering, strict matching, fan-out, cancellation, and cache safety."""

import asyncio

import pytest

from app.models import SubtitleRelease
from app.services.sync.cache import ReferenceDiskCache
from app.services.sync.external_strategy import ExternalExactStrategy
from app.services.sync.fanout import ReferenceFanoutLimiter
from app.services.sync.query import ReferenceQuery


def _payload(label="reference"):
    return (
        "\n\n".join(
            f"{n}\n00:00:{n:02d},000 --> 00:00:{n + 1:02d},000\n{label} line {n}"
            for n in range(1, 180)
        )
        + "\n"
    ).encode()


def _release(name="Show.S01E02.1080p.BluRay.x264-GRP.srt", ref="https://signed.test/ref?token=secret"):
    return SubtitleRelease(
        release_name=name,
        download_url=ref,
        provider="reference",
        lang="eng",
    )


def _query(**updates):
    values = {
        "imdb_id": "tt1",
        "media_type": "series",
        "season": 1,
        "episode": 2,
        "target_filename": "Show.S01E02.1080p.BluRay.x264-GRP.mkv",
        "languages": ("eng",),
        "api_keys": {"subdl": "transient-key", "subsource": "transient-key-2"},
    }
    values.update(updates)
    return ReferenceQuery(**values)


class FakeProvider:
    def __init__(self, name, releases=(), payload=None, search=None, download=None):
        self.name = name
        self.releases = list(releases)
        self.payload = payload
        self.search_hook = search
        self.download_hook = download
        self.search_calls = []
        self.download_calls = []

    async def search_subtitles(self, **kwargs):
        self.search_calls.append(kwargs)
        if self.search_hook:
            return await self.search_hook(**kwargs)
        return self.releases

    async def download_archive(self, download_ref, api_key=None):
        self.download_calls.append((download_ref, api_key))
        if self.download_hook:
            return await self.download_hook(download_ref, api_key)
        return self.payload


def _strategy(tmp_path, **providers):
    return ExternalExactStrategy(
        **providers,
        timeout=2.0,
        provider_timeout=1.0,
        min_bytes=100,
        cache=ReferenceDiskCache(tmp_path, min_bytes=100),
    )


@pytest.mark.asyncio
async def test_provider_tie_order_is_deterministic_and_downloads_are_sequential(tmp_path):
    order = []
    active_downloads = 0
    max_active = 0

    async def make_download(name):
        async def download(url, api_key):
            nonlocal active_downloads, max_active
            order.append(name)
            active_downloads += 1
            max_active = max(max_active, active_downloads)
            await asyncio.sleep(0.01)
            active_downloads -= 1
            return _payload(name)

        return download

    release = _release()
    providers = {
        "subdl_provider": FakeProvider("subdl", [release], download=await make_download("subdl")),
        "subsource_provider": FakeProvider("subsource", [release], download=await make_download("subsource")),
        "podnapisi_provider": FakeProvider("podnapisi", [release], download=await make_download("podnapisi")),
    }
    result = await _strategy(tmp_path, **providers).resolve_with_provenance(_query())
    assert result.text and "subdl line" in result.text
    assert order == ["subdl"]
    assert max_active == 1


@pytest.mark.asyncio
async def test_failed_download_falls_through_to_next_strict_candidate(tmp_path):
    rejected = _release(ref="https://signed.test/subdl?token=never-logged")
    valid = _release(ref="https://signed.test/subsource?token=never-logged")
    subdl = FakeProvider("subdl", [rejected], payload=None)
    subsource = FakeProvider("subsource", [valid], payload=_payload("subsource"))
    strategy = _strategy(tmp_path, subdl_provider=subdl, subsource_provider=subsource)
    result = await strategy.resolve_with_provenance(_query())
    assert result.text and "subsource line" in result.text
    assert len(subdl.download_calls) == 1
    assert len(subsource.download_calls) == 1


@pytest.mark.asyncio
async def test_strict_matching_rejects_wrong_source_before_download(tmp_path):
    wrong = _release("Show.S01E02.1080p.WEB-DL.x264-GRP.srt")
    subdl = FakeProvider("subdl", [wrong], payload=_payload())
    result = await _strategy(tmp_path, subdl_provider=subdl).resolve_with_provenance(_query())
    assert result.text is None
    assert subdl.download_calls == []


@pytest.mark.asyncio
async def test_series_season_and_episode_filter_before_download(tmp_path):
    wrong_season = _release("Show.S02E02.1080p.BluRay.x264-GRP.srt", "https://x.test/wrong-season")
    wrong_episode = _release("Show.S01E03.1080p.BluRay.x264-GRP.srt", "https://x.test/wrong-episode")
    correct = _release("Show.S01E02.1080p.BluRay.x264-GRP.srt", "https://x.test/correct")
    provider = FakeProvider("subdl", [wrong_season, wrong_episode, correct], payload=_payload())
    result = await _strategy(tmp_path, subdl_provider=provider).resolve_with_provenance(_query())
    assert result.text
    assert [ref for ref, _ in provider.download_calls] == ["https://x.test/correct"]
    assert provider.search_calls[0]["season"] == 1
    assert provider.search_calls[0]["episode"] == 2


@pytest.mark.asyncio
async def test_provider_failure_does_not_cancel_slower_valid_search(tmp_path):
    slow_release = _release(ref="https://x.test/slower")

    async def broken(**kwargs):
        raise RuntimeError("provider transport error")

    async def slow(**kwargs):
        await asyncio.sleep(0.03)
        return [slow_release]

    failed = FakeProvider("subdl", search=broken)
    slower = FakeProvider("subsource", search=slow, payload=_payload("slower"))
    result = await _strategy(
        tmp_path, subdl_provider=failed, subsource_provider=slower
    ).resolve_with_provenance(_query())
    assert result.text and "slower line" in result.text
    assert len(slower.download_calls) == 1


@pytest.mark.asyncio
async def test_provider_timeout_falls_through(tmp_path):
    async def slow_search(**kwargs):
        await asyncio.sleep(0.2)
        return [_release()]

    slow = FakeProvider("subdl", search=slow_search)
    quick = FakeProvider("subsource", [_release()], payload=_payload("quick"))
    strategy = ExternalExactStrategy(
        subdl_provider=slow,
        subsource_provider=quick,
        timeout=1.0,
        provider_timeout=0.02,
        min_bytes=100,
        cache=ReferenceDiskCache(tmp_path, min_bytes=100),
    )
    result = await strategy.resolve_with_provenance(_query())
    assert result.text and "quick line" in result.text
    assert slow.download_calls == []


@pytest.mark.asyncio
async def test_global_lookup_cap_is_shared_across_strategy_instances(tmp_path):
    limiter = ReferenceFanoutLimiter(2)
    started = 0
    max_started = 0
    six_started = asyncio.Event()
    release = asyncio.Event()

    async def blocked_search(**kwargs):
        nonlocal started, max_started
        started += 1
        max_started = max(max_started, started)
        if started >= 6:
            six_started.set()
        await release.wait()
        started -= 1
        return []

    providers = {
        "subdl_provider": FakeProvider("subdl", search=blocked_search),
        "subsource_provider": FakeProvider("subsource", search=blocked_search),
        "podnapisi_provider": FakeProvider("podnapisi", search=blocked_search),
    }
    strategies = [
        _strategy(tmp_path / str(i), limiter=limiter, **providers) for i in range(5)
    ]
    tasks = [asyncio.create_task(strategy.resolve(_query())) for strategy in strategies]
    await asyncio.wait_for(six_started.wait(), timeout=1)
    await asyncio.sleep(0.03)
    assert max_started == 6  # two admitted lookups, three fixed provider searches each
    release.set()
    await asyncio.gather(*tasks)


@pytest.mark.asyncio
async def test_cancellation_drains_searches_and_releases_global_permit(tmp_path):
    limiter = ReferenceFanoutLimiter(1)
    started = asyncio.Event()
    stopped = asyncio.Event()
    continue_search = asyncio.Event()
    calls = 0

    async def search(**kwargs):
        nonlocal calls
        calls += 1
        started.set()
        try:
            await continue_search.wait()
        finally:
            stopped.set()
        return []

    provider = FakeProvider("subdl", search=search)
    strategy = _strategy(tmp_path, limiter=limiter, subdl_provider=provider)
    query = ReferenceQuery(imdb_id="tt1")
    task = asyncio.create_task(strategy.resolve(query))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert stopped.is_set()

    continue_search.set()
    assert await asyncio.wait_for(strategy.resolve(query), timeout=1) is None
    assert calls == 2


@pytest.mark.asyncio
async def test_failure_does_not_poison_later_successful_reference(tmp_path):
    provider = FakeProvider("subdl", [], payload=_payload())
    cache = ReferenceDiskCache(tmp_path, min_bytes=100)
    strategy = ExternalExactStrategy(
        subdl_provider=provider, timeout=1, provider_timeout=0.5,
        min_bytes=100, cache=cache,
    )
    assert await strategy.resolve(_query()) is None
    assert list(tmp_path.iterdir()) == []

    provider.releases = [_release()]
    result = await strategy.resolve_with_provenance(_query())
    assert result.text
    assert list(tmp_path.glob("*.srt"))
