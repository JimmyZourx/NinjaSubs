"""Hash-exact English reference lookup through OpenSubtitles."""

from __future__ import annotations

import asyncio
import logging
import zipfile
from io import BytesIO
from typing import TYPE_CHECKING, Any

from app.extractor import MAX_SUBTITLE_ENTRY_BYTES, extract_srt_from_zip, transcode_to_utf8
from app.services.sync.cache import ReferenceDiskCache
from app.services.sync.query import ResolvedReference

if TYPE_CHECKING:
    from app.services.sync.query import ReferenceQuery

logger = logging.getLogger(__name__)


class HashExactStrategy:
    """Resolve an English subtitle only when the upstream confirms video-hash match."""

    name = "opensubtitles-hash"

    def __init__(
        self,
        opensubtitles_provider: Any | None = None,
        *,
        timeout: float = 12.0,
        min_bytes: int = 5120,
        cache: ReferenceDiskCache | None = None,
    ) -> None:
        self._provider = opensubtitles_provider
        self.timeout = max(0.1, timeout)
        self.min_bytes = max(0, min_bytes)
        self.cache = cache if cache is not None else ReferenceDiskCache(min_bytes=min_bytes)

    async def resolve(self, query: ReferenceQuery) -> str | None:
        return (await self.resolve_with_provenance(query)).text

    async def resolve_with_provenance(self, query: ReferenceQuery) -> ResolvedReference:
        cached = self.cache.get(query)
        if cached is not None:
            return cached
        if self._provider is None or not (query.video_hash or "").strip():
            return ResolvedReference(None)
        try:
            text = await asyncio.wait_for(self._run(query), timeout=self.timeout)
        except Exception as exc:
            logger.warning("[reference] hash strategy unavailable: %s", type(exc).__name__)
            return ResolvedReference(None)
        return ResolvedReference(text, kind="hash") if text else ResolvedReference(None)

    async def _run(self, query: ReferenceQuery) -> str | None:
        provider = self._provider
        if provider is None:
            return None
        api_key = query.api_keys.get("opensubtitles") or None
        releases = await provider.search_subtitles(
            imdb_id=query.imdb_id,
            is_series=query.is_series,
            season=query.season,
            episode=query.episode,
            title=query.title,
            year=query.year,
            api_key=api_key,
            languages=["eng"],
            video_hash=query.video_hash,
            video_size=query.video_size,
        )
        matches = [
            release for release in releases or []
            if bool(getattr(release, "is_hash_match", False))
            and str(getattr(release, "lang", "") or "").lower().startswith("en")
        ]
        matches.sort(key=lambda release: bool(getattr(release, "hearing_impaired", False)))
        for release in matches:
            raw = await provider.download_archive(release.download_url, api_key=api_key)
            if not raw:
                continue
            try:
                if zipfile.is_zipfile(BytesIO(raw)):
                    decoded = extract_srt_from_zip(
                        raw, target_filename=release.release_name,
                        season=query.season, episode=query.episode,
                    )
                else:
                    if len(raw) > MAX_SUBTITLE_ENTRY_BYTES:
                        continue
                    decoded = transcode_to_utf8(raw)
            except Exception as exc:
                logger.warning("[reference] hash candidate rejected: %s", type(exc).__name__)
                continue
            if len(decoded) <= self.min_bytes:
                continue
            text = decoded.decode("utf-8", "replace")
            self.cache.set(query, "opensubtitles", text, kind="hash", candidate=release.release_name)
            return text
        return None
