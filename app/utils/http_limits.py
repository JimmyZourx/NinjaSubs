"""Bounded HTTP download helpers for resource exhaustion protection."""

import logging
from urllib.parse import urljoin, urlparse

import httpx

logger = logging.getLogger(__name__)

MAX_UPSTREAM_DOWNLOAD_BYTES = 10 * 1024 * 1024
MAX_HTML_RESPONSE_BYTES = 2 * 1024 * 1024


def safe_url_for_logging(url: str) -> str:
    """Return a sanitized URL for logging — host only, no query params or path details."""
    try:
        parsed = urlparse(url)
        host = parsed.hostname or "<unknown-host>"
        if parsed.port is not None:
            host = f"{host}:{parsed.port}"
        return f"{parsed.scheme}://{host}"
    except Exception:
        return "<redacted>"


def _origin(url: str) -> tuple[str, str, int | None]:
    parsed = urlparse(url)
    port = parsed.port
    if port is None:
        port = 443 if parsed.scheme.lower() == "https" else 80
    return parsed.scheme.lower(), (parsed.hostname or "").lower(), port


def _strip_secret_headers(headers: dict[str, str]) -> dict[str, str]:
    secret_names = {"authorization", "proxy-authorization", "cookie", "x-api-key", "api-key"}
    return {key: value for key, value in headers.items() if key.lower() not in secret_names}


async def bounded_download_bytes(
    client: httpx.AsyncClient,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    params: dict | None = None,
    timeout: float | None = None,
    max_bytes: int = MAX_UPSTREAM_DOWNLOAD_BYTES,
    follow_redirects: bool = True,
) -> tuple[int, bytes]:
    """Stream download with hard byte limit.

    Performs exactly ONE streaming GET request.
    Returns (status_code, data). On non-200, returns (status_code, b"") without
    reading the response body. On success, returns (200, data).

    Returns an empty body if a size limit is exceeded. Redirects are followed
    explicitly so provider secrets are removed before cross-origin requests.
    """
    req_headers = dict(headers or {})
    current_url = url
    current_params = params
    initial_origin = _origin(url)
    try:
        for redirect_count in range(11):
            async with client.stream(
                "GET",
                current_url,
                headers=req_headers,
                params=current_params,
                timeout=timeout,
                follow_redirects=False,
            ) as resp:
                status = resp.status_code
                if status in (301, 302, 303, 307, 308) and follow_redirects:
                    location = resp.headers.get("location")
                    if not location or redirect_count == 10:
                        return status, b""
                    next_url = urljoin(current_url, location)
                    if _origin(next_url) != initial_origin:
                        req_headers = _strip_secret_headers(req_headers)
                    current_url = next_url
                    current_params = None
                    continue

                if status != 200:
                    return status, b""

                content_length = resp.headers.get("content-length")
                if content_length:
                    try:
                        if int(content_length) > max_bytes:
                            logger.warning(
                                "DownloadTooLarge: Content-Length %s > %s for %s",
                                content_length, max_bytes, safe_url_for_logging(current_url)
                            )
                            return status, b""
                    except ValueError:
                        pass

                chunks = []
                total = 0
                async for chunk in resp.aiter_bytes(chunk_size=64 * 1024):
                    total += len(chunk)
                    if total > max_bytes:
                        logger.warning(
                            "DownloadTooLarge: streamed %s > %s for %s",
                            total, max_bytes, safe_url_for_logging(current_url)
                        )
                        return status, b""
                    chunks.append(chunk)
                return status, b"".join(chunks)
        return 0, b""
    except httpx.TimeoutException:
        logger.warning("Bounded download timeout for %s", safe_url_for_logging(url))
        return 0, b""
    except Exception as e:
        logger.warning("Bounded download error for %s: %s", safe_url_for_logging(url), type(e).__name__)
        return 0, b""


async def bounded_fetch_text(
    client: httpx.AsyncClient,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    params: dict | None = None,
    timeout: float | None = None,
    max_bytes: int = MAX_HTML_RESPONSE_BYTES,
    follow_redirects: bool = True,
) -> tuple[int, str]:
    """Bounded HTML/text fetch with size limit.

    Performs exactly ONE streaming GET request.
    Returns (status_code, text). On non-200, returns (status_code, "") without
    reading the response body. On success, returns (200, text).

    Returns (0, "") on timeout/error.
    """
    try:
        async with client.stream(
            "GET",
            url,
            headers=headers,
            params=params,
            timeout=timeout,
            follow_redirects=follow_redirects,
        ) as resp:
            status = resp.status_code
            if status != 200:
                return status, ""

            content_length = resp.headers.get("content-length")
            if content_length:
                try:
                    if int(content_length) > max_bytes:
                        logger.warning(
                            "HTMLTooLarge: Content-Length %s > %s for %s",
                            content_length, max_bytes, safe_url_for_logging(url)
                        )
                        return status, ""
                except ValueError:
                    pass

            chunks = []
            total = 0
            async for chunk in resp.aiter_bytes(chunk_size=64 * 1024):
                total += len(chunk)
                if total > max_bytes:
                    logger.warning(
                        "HTMLTooLarge: streamed %s > %s for %s",
                        total, max_bytes, safe_url_for_logging(url)
                    )
                    return status, ""
                chunks.append(chunk)

            data = b"".join(chunks)
            encoding = resp.encoding or "utf-8"
            try:
                return status, data.decode(encoding, errors="replace")
            except Exception:
                return status, data.decode("utf-8", errors="replace")
    except httpx.TimeoutException:
        logger.warning("Bounded fetch timeout for %s", safe_url_for_logging(url))
        return 0, ""
    except Exception as e:
        logger.warning("Bounded fetch error for %s: %s", safe_url_for_logging(url), type(e).__name__)
        return 0, ""


def safe_headers_for_cdn(base_headers: dict[str, str] | None = None) -> dict[str, str]:
    """Return headers safe to send to a CDN redirect URL.

    Strips provider-secret headers (X-API-Key, Authorization) and keeps
    only generic headers like User-Agent, Accept, Referer.
    """
    safe_keys = {"user-agent", "accept", "accept-language", "referer", "accept-encoding"}
    out: dict[str, str] = {}
    for k, v in (base_headers or {}).items():
        if k.lower() in safe_keys:
            out[k] = v
    return out
