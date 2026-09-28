"""Bounded Podnapisi provider used only as an English AutoSync reference."""

from __future__ import annotations

import html
import logging
import re

from app.config import settings
from app.models import SubtitleRelease
from app.providers.base import BaseSubtitleProvider
from app.utils.http_limits import (
    bounded_download_bytes,
    bounded_fetch_text,
    is_allowed_provider_url,
    safe_url_for_logging,
)
from app.utils.language import normalize_to_iso639_2

logger = logging.getLogger("uvicorn.error")

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)
_ROW_RE = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.I | re.S)
_PID_RE = (
    re.compile(r'data-pid=["\'](\d+)["\']', re.I),
    re.compile(r'href=["\']/subtitles/(\d+)(?:/download)?/?["\']', re.I),
)
_LANG_RE = re.compile(r'<abbr\b[^>]*title=["\']([^"\']+)["\']', re.I)
_FLAG_RE = re.compile(r"flag-([a-z]{2,3})\b", re.I)
_RELEASE_RE = (
    re.compile(r'<span\b[^>]*class=["\'][^"\']*release[^"\']*["\'][^>]*>(.*?)</span>', re.I | re.S),
    re.compile(r'<td\b[^>]*class=["\'][^"\']*release[^"\']*["\'][^>]*>(.*?)</td>', re.I | re.S),
)
_TITLE_RE = re.compile(r'<a\b[^>]*class=["\'][^"\']*title[^"\']*["\'][^>]*>(.*?)</a>', re.I | re.S)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_PODNAPISI_HOSTS = frozenset({"podnapisi.net"})  # includes only this domain and its subdomains


def _clean(fragment: str) -> str:
    return _WS_RE.sub(" ", html.unescape(_TAG_RE.sub("", fragment))).strip()


class PodnapisiProvider(BaseSubtitleProvider):
    """Search and bounded-download English reference subtitles from Podnapisi."""

    name = "podnapisi"
    BASE_URL = "https://www.podnapisi.net"
    SEARCH_URL = f"{BASE_URL}/subtitles/search/old"

    def _headers(self) -> dict[str, str]:
        return {
            "User-Agent": _UA,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": f"{self.BASE_URL}/",
        }

    @staticmethod
    def _row_pid(row: str) -> str | None:
        for pattern in _PID_RE:
            match = pattern.search(row)
            if match:
                return match.group(1)
        return None

    @staticmethod
    def _row_language(row: str) -> str:
        match = _LANG_RE.search(row)
        if match:
            return normalize_to_iso639_2(match.group(1), default="eng")
        match = _FLAG_RE.search(row)
        if match:
            return normalize_to_iso639_2(match.group(1), default="eng")
        return "eng"

    @staticmethod
    def _row_release(row: str) -> str:
        for pattern in _RELEASE_RE:
            match = pattern.search(row)
            if match:
                release = _clean(match.group(1))
                if release:
                    return release
        match = _TITLE_RE.search(row)
        return _clean(match.group(1)) if match else ""

    async def search_subtitles(
        self,
        imdb_id: str,
        is_series: bool = False,
        season: int | None = None,
        episode: int | None = None,
        title: str | None = None,
        year: int | None = None,
        api_key: str | None = None,
        languages: list[str] | None = None,
        exclude_hi: bool = False,
        **kwargs,
    ) -> list[SubtitleRelease]:
        """Search for English only; Podnapisi is a reference source, not a listing provider."""
        keywords = (title or "").strip() or str(imdb_id or "").split(":", 1)[0].strip()
        if not keywords:
            return []
        params = {
            "keywords": keywords,
            "movie_type": "tv-series" if is_series else "movie",
            "language": "en",
        }
        if is_series and season is not None:
            params["seasons"] = str(season)
        if is_series and episode is not None:
            params["episodes"] = str(episode)
        if year:
            params["year"] = str(year)
        clean_imdb = str(imdb_id or "").split(":", 1)[0].strip()
        if re.fullmatch(r"tt\d+", clean_imdb):
            params["imdb"] = clean_imdb

        status, page = await bounded_fetch_text(
            self.client,
            self.SEARCH_URL,
            headers=self._headers(),
            params=params,
            timeout=settings.UPSTREAM_TIMEOUT,
            max_bytes=2 * 1024 * 1024,
            follow_redirects=True,
            allowed_hosts=_PODNAPISI_HOSTS,
        )
        if status != 200 or not page:
            logger.info("[Podnapisi] Search unavailable (HTTP %s)", status)
            return []

        results: list[SubtitleRelease] = []
        seen_ids: set[str] = set()
        for row_match in _ROW_RE.finditer(page):
            row = row_match.group(0)
            pid = self._row_pid(row)
            release = self._row_release(row)
            if not pid or pid in seen_ids or not release:
                continue
            if self._row_language(row) != "eng":
                continue
            seen_ids.add(pid)
            results.append(
                SubtitleRelease(
                    release_name=release[:512],
                    download_url=f"{self.BASE_URL}/subtitles/{pid}/download",
                    provider=self.name,
                    format="srt",
                    lang="eng",
                )
            )
        logger.info("[Podnapisi] Found %d English reference(s)", len(results))
        return results

    async def download_archive(self, download_ref: str, api_key: str | None = None) -> bytes | None:
        """Download a Podnapisi ZIP under the binary cap and trusted-host policy."""
        reference = str(download_ref or "").strip()
        if re.fullmatch(r"\d+", reference):
            url = f"{self.BASE_URL}/subtitles/{reference}/download"
        elif re.fullmatch(r"/subtitles/\d+/download/?", reference):
            url = f"{self.BASE_URL}{reference.rstrip('/')}"
        elif is_allowed_provider_url(reference, _PODNAPISI_HOSTS):
            url = reference
        else:
            logger.warning("[Podnapisi] Download rejected an unapproved host")
            return None

        status, payload = await bounded_download_bytes(
            self.client,
            url,
            headers=self._headers(),
            timeout=settings.UPSTREAM_TIMEOUT,
            max_bytes=10 * 1024 * 1024,
            follow_redirects=True,
            allowed_hosts=_PODNAPISI_HOSTS,
        )
        if status != 200 or not payload:
            logger.warning(
                "[Podnapisi] Download unavailable (HTTP %s) from %s",
                status,
                safe_url_for_logging(url),
            )
            return None
        return payload
