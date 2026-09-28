"""YIFYSubtitles (yifysubtitles.ch) provider integration.

YIFYSubtitles is a movies-only subtitle mirror. Every subtitle for a movie is
listed server-side on ``/movie-imdb/{imdb_id}`` and each row links to a
``/subtitle/{slug}.zip`` archive. Downloads are fronted by Cloudflare and
require the ``PHPSESSID`` cookie obtained from the listing page plus a Referer
header, so the shared HTTP client's cookie jar is reused and a warm-up request
is performed on a 403.
"""

import html
import logging
import re

import httpx

from app.config import settings
from app.models import SubtitleRelease
from app.providers.base import BaseSubtitleProvider
from app.utils.language import normalize_to_iso639_2

logger = logging.getLogger("uvicorn.error")

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

_ROW_REGEX = re.compile(r'<tr\s+data-id="(\d+)">(.*?)</tr>', re.IGNORECASE | re.DOTALL)
_LANG_REGEX = re.compile(r'<span class="sub-lang">([^<]+)</span>', re.IGNORECASE)
_LINK_REGEX = re.compile(
    r'<a\s+href="(/subtitles/[^"]+)"[^>]*>(.*?)</a>', re.IGNORECASE | re.DOTALL
)
_UPLOADER_REGEX = re.compile(
    r'<td[^>]*class="uploader-cell"[^>]*>\s*<a[^>]*>([^<]*)</a>', re.IGNORECASE
)
_RATING_REGEX = re.compile(
    r'<td[^>]*class="rating-cell"[^>]*>\s*<span[^>]*class="label"[^>]*>([^<]*)</span>',
    re.IGNORECASE,
)
_TAG_REGEX = re.compile(r"<[^>]+>")


class YifysubtitlesProvider(BaseSubtitleProvider):
    """Subtitle provider implementation for yifysubtitles.ch (movies only)."""

    name = "yifysubtitles"
    BASE_URL = "https://yifysubtitles.ch"

    def _headers(self, referer: str | None = None) -> dict[str, str]:
        headers = {
            "User-Agent": _UA,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        headers["Referer"] = referer or f"{self.BASE_URL}/"
        return headers

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
        """Query YIFYSubtitles by IMDb ID (movies only; series are unsupported)."""
        from app.utils.http_limits import bounded_fetch_text

        if is_series:
            return []

        clean_imdb = str(imdb_id or "").split(":")[0].strip()
        if not re.match(r"^tt\d+$", clean_imdb):
            return []

        page_url = f"{self.BASE_URL}/movie-imdb/{clean_imdb}"
        try:
            status, page_html = await bounded_fetch_text(
                self.client,
                page_url,
                headers=self._headers(),
                timeout=settings.UPSTREAM_TIMEOUT,
            )
        except httpx.TimeoutException:
            logger.warning(f"[YIFYSubtitles] Search timed out for {clean_imdb}")
            return []
        except Exception as e:
            logger.warning(f"[YIFYSubtitles] Search error for {clean_imdb}: {e}")
            return []

        if status != 200 or "Page not found" in page_html:
            logger.info(
                f"[YIFYSubtitles] No listing for {clean_imdb} (HTTP {status})"
            )
            return []

        wanted = {normalize_to_iso639_2(lang) for lang in (languages or []) if lang}
        results: list[SubtitleRelease] = []
        seen_slugs: set[str] = set()

        for row_match in _ROW_REGEX.finditer(page_html):
            row = row_match.group(2)

            lang_match = _LANG_REGEX.search(row)
            link_match = _LINK_REGEX.search(row)
            if not link_match:
                continue

            lang_name = lang_match.group(1).strip() if lang_match else ""
            lang_iso = normalize_to_iso639_2(lang_name, default="")
            if wanted and lang_iso not in wanted:
                continue

            slug_path = link_match.group(1).strip()
            if slug_path in seen_slugs:
                continue
            seen_slugs.add(slug_path)
            slug = slug_path[len("/subtitles/") :]

            # Some rows append every compatible release separated by <br />; keep only
            # the primary (first) release name so it stays a clean, matchable string.
            anchor_html = re.sub(r"(?i)<br\s*/?>", "\n", link_match.group(2))
            anchor_text = html.unescape(_TAG_REGEX.sub("", anchor_html))
            lines = [line.strip() for line in anchor_text.splitlines() if line.strip()]
            release_name = lines[0] if lines else ""
            release_name = re.sub(r"(?i)^subtitle\s*", "", release_name).strip()
            if not release_name:
                release_name = slug

            uploader_match = _UPLOADER_REGEX.search(row)
            uploader = html.unescape(uploader_match.group(1)).strip() if uploader_match else ""

            rating_match = _RATING_REGEX.search(row)
            is_hi = False
            if rating_match:
                try:
                    is_hi = float(html.unescape(rating_match.group(1)).strip() or 0) < 0
                except (ValueError, TypeError):
                    is_hi = False

            results.append(
                SubtitleRelease(
                    release_name=release_name,
                    download_url=f"{self.BASE_URL}/subtitle/{slug}.zip",
                    provider=self.name,
                    format="srt",
                    hearing_impaired=is_hi,
                    lang=lang_iso or "ara",
                    uploader=uploader,
                )
            )

        logger.info(
            f"[YIFYSubtitles] Found {len(results)} subtitles for {clean_imdb} "
            f"(Langs: {languages})"
        )
        return results

    async def download_archive(self, download_ref: str, api_key: str | None = None) -> bytes | None:
        """Download a subtitle .zip archive, handling Cloudflare session warm-up."""
        from app.utils.http_limits import bounded_download_bytes, bounded_fetch_text

        url = download_ref
        if not url.startswith("http"):
            url = f"{self.BASE_URL}/{url.lstrip('/')}"

        headers = dict(self._headers())
        headers["Accept"] = "application/octet-stream,application/zip,*/*"

        try:
            # Single streaming request
            status, data = await bounded_download_bytes(
                self.client,
                url,
                headers=headers,
                timeout=settings.UPSTREAM_TIMEOUT,
            )
            if status == 200 and data:
                return data

            # Cloudflare may require a fresh session cookie; warm up and retry once.
            if status in (403, 503):
                logger.info("[YIFYSubtitles] Warming up session after HTTP %s", status)
                try:
                    # Use bounded fetch for warm-up to avoid unbounded read
                    await bounded_fetch_text(
                        self.client,
                        f"{self.BASE_URL}/",
                        headers=self._headers(),
                        timeout=settings.UPSTREAM_TIMEOUT,
                    )
                except Exception:
                    pass
                status, data = await bounded_download_bytes(
                    self.client,
                    url,
                    headers=headers,
                    timeout=settings.UPSTREAM_TIMEOUT,
                )
                if status == 200 and data:
                    return data

            logger.warning(
                f"[YIFYSubtitles] Download returned HTTP {status}"
            )
            return None
        except httpx.TimeoutException:
            logger.warning("[YIFYSubtitles] Download timed out")
            return None
        except Exception as e:
            logger.warning(f"[YIFYSubtitles] Download error: {type(e).__name__}")
            return None
