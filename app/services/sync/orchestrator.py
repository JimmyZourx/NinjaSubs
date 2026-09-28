"""AutoSync gates, reference query construction, and strategy coordination."""

from __future__ import annotations

import asyncio
import hashlib
import re
from typing import Any

from app.config import settings
from app.services.sync.matching import (
    _edition_tags,
    _release_group,
    _source_kind,
    _sources_compatible,
    is_informative_release_name,
)
from app.services.sync.query import ReferenceQuery, ResolvedReference


def _normalize_fingerprint(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def build_synced_cache_key(
    meta: dict,
    target_id: str,
    content_hash: str | None = None,
    decision: str | None = None,
) -> str:
    """Construct a media- and decision-scoped key without persisting credentials."""
    imdb_id = str(meta.get("imdb_id") or "unknown").strip() or "unknown"
    season, episode = meta.get("season"), meta.get("episode")
    season_ep = f"s{season}e{episode}" if season is not None and episode is not None else "movie"
    video_hash = _normalize_fingerprint(meta.get("video_hash"))
    stable_identity = video_hash or _normalize_fingerprint(meta.get("target_filename")) or _normalize_fingerprint(target_id)
    fingerprint = hashlib.sha256(stable_identity.encode("utf-8")).hexdigest()[:20]
    key = f"final_sub:{imdb_id}:{season_ep}:{fingerprint}:{target_id}"
    if decision:
        key += f":{decision}"
    if content_hash:
        key += f":{content_hash}"
    return key


def _parse_year(value: Any) -> int | None:
    try:
        return int(str(value).strip()) if str(value or "").strip() else None
    except (TypeError, ValueError):
        return None


class SyncOrchestrator:
    """Apply server/user/language gates and coordinate injected reference strategies.

    Stage 1 intentionally has no synchronizer attached, so enabled requests
    safely return their original bytes until the execution stage is ported.
    """

    def __init__(
        self,
        *,
        hash_strategy: Any | None = None,
        embedded_strategy: Any | None = None,
        external_strategy: Any | None = None,
        sync_service: Any | None = None,
        sync_cache: Any | None = None,
    ) -> None:
        self._hash_strategy = hash_strategy
        self._embedded_strategy = embedded_strategy
        self._external_strategy = external_strategy
        self._sync_service = sync_service
        self._sync_cache = sync_cache
        self._inflight: dict[str, asyncio.Future[bytes]] = {}
        self._inflight_lock = asyncio.Lock()

    def _strategies(self) -> list[tuple[str, Any]]:
        return [
            ("hash-exact", self._hash_strategy),
            ("embedded", self._embedded_strategy),
            ("external exact-match", self._external_strategy),
        ]

    def _build_query(self, meta: dict) -> ReferenceQuery:
        return ReferenceQuery(
            imdb_id=str(meta.get("imdb_id") or ""),
            target_filename=meta.get("target_filename") or meta.get("release_name"),
            media_type=str(meta.get("media_type") or "movie"),
            title=meta.get("title"),
            year=_parse_year(meta.get("year")),
            video_hash=meta.get("video_hash"),
            video_size=meta.get("video_size"),
            stream_url=meta.get("stream_url"),
            season=meta.get("season"),
            episode=meta.get("episode"),
            api_keys={
                "subdl": meta.get("subdl_key") or "",
                "subsource": meta.get("subsource_key") or "",
                "opensubtitles": meta.get("opensubtitles_key") or "",
            },
            languages=("eng",),
        )

    async def evaluate_and_sync(
        self,
        sub_bytes: bytes,
        meta: dict,
        target_id: str,
        auto_sync: bool = False,
    ) -> bytes:
        """Return original bytes unless AutoSync is enabled and executable sync is installed."""
        if not settings.AUTOSYNC_ENABLED or not auto_sync:
            return sub_bytes
        if not str(meta.get("lang") or "").lower().startswith("ar"):
            return sub_bytes
        # Foundation-only build: do not fan out to providers until the resolver
        # and synchronization execution stage are installed together.
        if self._sync_service is None:
            return sub_bytes
        key = self._flight_key(meta, target_id, sub_bytes)
        async with self._inflight_lock:
            future = self._inflight.get(key)
            if future is None:
                future = asyncio.ensure_future(self._execute(sub_bytes, meta, target_id))
                self._inflight[key] = future
                future.add_done_callback(lambda done: self._finish_flight(key, done))
        return await asyncio.shield(future)

    async def resolve_reference(self, meta: dict) -> ResolvedReference:
        """Ask the configured foundation strategies for the first reference."""
        query = self._build_query(meta)
        for _name, strategy in self._strategies():
            if strategy is None:
                continue
            try:
                result = await strategy.resolve_with_provenance(query)
            except Exception:
                continue
            if result.text:
                return result
        return ResolvedReference(None)

    async def _execute(self, sub_bytes: bytes, meta: dict, target_id: str) -> bytes:
        if self._sync_service is None:
            return sub_bytes
        resolved = await self.resolve_reference(meta)
        if not resolved.text:
            return sub_bytes
        from app.extractor import transcode_to_utf8

        target = transcode_to_utf8(sub_bytes).decode("utf-8", "replace")
        synced = await self._sync_service.sync_async(
            target,
            resolved.text,
            decision_kind=resolved.kind,
            is_series=self._build_query(meta).is_series,
            source_confirmed=resolved.bluray_match,
        )
        return synced.encode("utf-8") if synced else sub_bytes

    @staticmethod
    def _flight_key(meta: dict, target_id: str, sub_bytes: bytes) -> str:
        digest = hashlib.sha256(sub_bytes).hexdigest()[:16]
        return build_synced_cache_key(meta, target_id, content_hash=digest)

    def _finish_flight(self, key: str, future: asyncio.Future[bytes]) -> None:
        if self._inflight.get(key) is future:
            del self._inflight[key]
        if not future.cancelled():
            future.exception()

    async def close(self) -> None:
        pending = list(self._inflight.values())
        for future in pending:
            future.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)


__all__ = [
    "SyncOrchestrator",
    "build_synced_cache_key",
    "_edition_tags",
    "_release_group",
    "_source_kind",
    "_sources_compatible",
    "is_informative_release_name",
]
