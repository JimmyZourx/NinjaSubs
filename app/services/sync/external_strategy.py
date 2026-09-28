"""Bounded external reference acquisition for SubDL, SubSource, and Podnapisi."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from app.config import settings
from app.services.sync.cache import ReferenceDiskCache
from app.services.sync.decode import decode_payload
from app.services.sync.fanout import ReferenceFanoutLimiter, reference_fanout_limiter
from app.services.sync.matching import _source_kind, _title_from_filename, is_retail_disc_source
from app.services.sync.query import ReferenceQuery, ResolvedReference
from app.services.sync.tree import ReferenceDecision, decide

logger = logging.getLogger(__name__)

_DEFAULT_LOOKUP_TIMEOUT = 8.0
_DEFAULT_PROVIDER_TIMEOUT = 6.0
_MIN_REFERENCE_BYTES = 5120


@dataclass(frozen=True)
class _ProviderSearch:
    name: str
    provider: Any
    api_key: str | None
    releases: list[Any]


class ExternalExactStrategy:
    """Search providers concurrently, select strictly, then download sequentially."""

    name = "external"
    PROVIDER_ORDER = ("subdl", "subsource", "podnapisi")

    def __init__(
        self,
        subdl_provider: Any | None = None,
        subsource_provider: Any | None = None,
        podnapisi_provider: Any | None = None,
        *,
        timeout: float = _DEFAULT_LOOKUP_TIMEOUT,
        provider_timeout: float = _DEFAULT_PROVIDER_TIMEOUT,
        min_bytes: int = _MIN_REFERENCE_BYTES,
        cache: ReferenceDiskCache | None = None,
        limiter: ReferenceFanoutLimiter | None = None,
    ) -> None:
        self._providers = {
            "subdl": subdl_provider,
            "subsource": subsource_provider,
            "podnapisi": podnapisi_provider,
        }
        self.timeout = max(0.1, timeout)
        self.provider_timeout = max(0.1, provider_timeout)
        self.min_bytes = max(0, min_bytes)
        self.cache = cache if cache is not None else ReferenceDiskCache(min_bytes=min_bytes)
        self.limiter = limiter if limiter is not None else reference_fanout_limiter

    async def resolve(self, query: ReferenceQuery) -> str | None:
        return (await self.resolve_with_provenance(query)).text

    async def resolve_with_provenance(self, query: ReferenceQuery) -> ResolvedReference:
        cached = self.cache.get(query)
        if cached is not None:
            return cached
        async with self.limiter.acquire():
            try:
                async with asyncio.timeout(self.timeout):
                    return await self._resolve_admitted(query)
            except TimeoutError:
                logger.warning("[reference] external lookup exceeded its deadline")
                return ResolvedReference(None)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("[reference] external lookup failed: %s", type(exc).__name__)
                return ResolvedReference(None)

    async def _resolve_admitted(self, query: ReferenceQuery) -> ResolvedReference:
        # The permit is held before child creation, so there are never more than
        # three provider-search tasks per globally admitted lookup.
        tasks: dict[str, asyncio.Task[list[Any]]] = {}
        try:
            async with asyncio.TaskGroup() as group:
                for name in self.PROVIDER_ORDER:
                    provider = self._providers[name]
                    if provider is not None:
                        tasks[name] = group.create_task(
                            self._search_provider(name, provider, query), name=f"reference-search-{name}"
                        )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("[reference] provider search group failed: %s", type(exc).__name__)

        # Search completion order never changes candidate tie order.
        candidates: list[_ProviderSearch] = []
        for name in self.PROVIDER_ORDER:
            task = tasks.get(name)
            provider = self._providers[name]
            if task is None or provider is None or task.cancelled():
                continue
            try:
                releases = task.result()
            except Exception as exc:
                logger.warning("[reference] %s search failed: %s", name, type(exc).__name__)
                continue
            key = query.api_keys.get(name) if name in ("subdl", "subsource") else None
            candidates.append(_ProviderSearch(name, provider, key, releases))

        strict = bool(getattr(settings, "SYNC_REQUIRE_EXACT_MATCH", True))
        selected: list[tuple[_ProviderSearch, ReferenceDecision]] = []
        for result in candidates:
            decision = decide(result.releases, query, strict=strict, provider_name=result.name)
            if decision.release is not None and decision.kind != "abort":
                selected.append((result, decision))

        # Candidate downloads are deliberately serialized; a failed/bad archive
        # falls through to the next strictly acceptable provider.
        for result, decision in selected:
            release = decision.release
            if release is None:
                continue
            try:
                async with asyncio.timeout(self.provider_timeout):
                    payload = await result.provider.download_archive(
                        release.download_url, api_key=result.api_key
                    )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("[reference] %s download failed: %s", result.name, type(exc).__name__)
                continue
            if not payload:
                continue
            decoded = decode_payload(
                payload,
                str(getattr(release, "release_name", "")),
                season=query.season,
                episode=query.episode,
                min_bytes=self.min_bytes,
            )
            if not decoded:
                logger.info("[reference] %s candidate could not be decoded", result.name)
                continue
            text = decoded.decode("utf-8", "replace")
            if len(text.encode("utf-8")) <= self.min_bytes:
                continue
            name = str(getattr(release, "release_name", ""))[:512]
            target_is_disc = is_retail_disc_source(_source_kind(query.target_filename))
            reference_is_disc = is_retail_disc_source(_source_kind(name))
            self.cache.set(
                query,
                result.name,
                text,
                kind=decision.kind,
                bluray_match=target_is_disc and reference_is_disc,
                candidate=name,
            )
            return ResolvedReference(
                text,
                kind=decision.kind,
                bluray_match=target_is_disc and reference_is_disc,
                candidate=name,
            )
        return ResolvedReference(None)

    async def _search_provider(self, name: str, provider: Any, query: ReferenceQuery) -> list[Any]:
        key = query.api_keys.get(name) if name in ("subdl", "subsource") else None

        async def search(episode: int | None) -> list[Any]:
            title = query.title
            if name == "podnapisi" and not title:
                title = _title_from_filename(query.target_filename)
            kwargs = {
                "imdb_id": query.imdb_id,
                "is_series": query.is_series,
                "season": query.season,
                "episode": episode,
                "title": title,
                "year": query.year,
                "languages": list(query.languages),
                "target_filename": query.target_filename if query.target_filename else None,
            }
            if name in ("subdl", "subsource"):
                kwargs["api_key"] = key
            try:
                async with asyncio.timeout(self.provider_timeout):
                    results = await provider.search_subtitles(**kwargs)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("[reference] %s search failed: %s", name, type(exc).__name__)
                return []
            return [
                result for result in (results or [])
                if str(getattr(result, "lang", "") or "").lower().startswith("en")
            ]

        episode_results = await search(query.episode)
        if not self._needs_season_enrichment(episode_results, query):
            return episode_results
        season_results = await search(None)
        seen = {getattr(release, "download_url", None) for release in episode_results}
        return episode_results + [
            release for release in season_results
            if getattr(release, "download_url", None) not in seen
        ]

    @staticmethod
    def _needs_season_enrichment(releases: list[Any], query: ReferenceQuery) -> bool:
        if query.season is None:
            return False
        target_source = _source_kind(query.target_filename)
        if target_source is None:
            return not releases
        return not any(
            _source_kind(str(getattr(release, "release_name", "") or "")) == target_source
            or {target_source, _source_kind(str(getattr(release, "release_name", "") or ""))}
            == {"bluray", "remux"}
            for release in releases
        )
