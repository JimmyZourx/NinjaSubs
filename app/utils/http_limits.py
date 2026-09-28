"""Bounded HTTP download helpers for resource exhaustion protection."""

import ipaddress
import logging
import socket
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

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
    secret_names = {
        "authorization", "proxy-authorization", "cookie", "x-api-key", "api-key",
        "x-subdl-api-key", "x-subsource-api-key", "opensubtitles-api-key",
    }
    return {
        key: value
        for key, value in headers.items()
        if key.lower() not in secret_names
        and "api-key" not in key.lower()
        and "apikey" not in key.lower()
    }


_SECRET_QUERY_NAMES = frozenset(
    {"api_key", "apikey", "api-key", "subdl_key", "subsource_key", "opensubtitles_key"}
)


def _strip_secret_query(url: str) -> str:
    parsed = urlparse(url)
    query = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key.lower() not in _SECRET_QUERY_NAMES
        and "api_key" not in key.lower()
        and "apikey" not in key.lower()
    ]
    return urlunparse(parsed._replace(query=urlencode(query, doseq=True)))


def _host_allowed(url: str, allowed_hosts: set[str] | frozenset[str] | None) -> bool:
    """Require HTTPS and a non-IP host in an explicit provider host allowlist."""
    if allowed_hosts is None:
        return True
    try:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower().rstrip(".")
        if parsed.scheme.lower() != "https" or not host or parsed.username or parsed.password:
            return False
        if parsed.port not in (None, 443):
            return False
        try:
            ipaddress.ip_address(host)
        except ValueError:
            pass
        else:
            return False
        for allowed in allowed_hosts:
            domain = allowed.lower().lstrip(".").rstrip(".")
            if host == domain or host.endswith(f".{domain}"):
                return True
        return False
    except ValueError:
        return False


def is_allowed_provider_url(url: str, allowed_hosts: set[str] | frozenset[str]) -> bool:
    """Syntactically validate an HTTPS provider URL before attaching credentials.

    The bounded request helper repeats this check and validates resolved DNS
    addresses immediately before making the request.
    """
    return _host_allowed(url, allowed_hosts)


def _public_dns_target(url: str) -> bool:
    """Reject names that resolve to non-public addresses before connecting."""
    try:
        host = urlparse(url).hostname
        if not host:
            return False
        try:
            address = ipaddress.ip_address(host)
            return address.is_global
        except ValueError:
            records = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
        addresses = {ipaddress.ip_address(record[4][0]) for record in records if record[4]}
        return bool(addresses) and all(address.is_global for address in addresses)
    except (OSError, ValueError):
        return False


async def _url_is_allowed(
    url: str, allowed_hosts: set[str] | frozenset[str] | None
) -> bool:
    """Run potentially blocking DNS checks off the event loop when allowlisting."""
    if not _host_allowed(url, allowed_hosts):
        return False
    if allowed_hosts is None:
        return True
    import asyncio

    return await asyncio.to_thread(_public_dns_target, url)


async def bounded_request_bytes(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    params: dict | None = None,
    json_body: dict | None = None,
    timeout: float | None = None,
    max_bytes: int = MAX_UPSTREAM_DOWNLOAD_BYTES,
    accepted_statuses: tuple[int, ...] = (200,),
    follow_redirects: bool = False,
    allowed_hosts: set[str] | frozenset[str] | None = None,
) -> tuple[int, bytes]:
    """Bound an arbitrary HTTP response and validate every redirect hop.

    Non-accepted status bodies are not iterated. If an allowlist is supplied,
    both the initial URL and every redirect target must be HTTPS on an
    approved provider host. Secrets are stripped on every cross-origin hop.
    """
    current_url = url
    current_method = method.upper()
    current_headers = dict(headers or {})
    current_params = params
    current_json = json_body
    if not await _url_is_allowed(current_url, allowed_hosts):
        logger.warning("Bounded request rejected an unapproved initial host")
        return 0, b""

    try:
        for redirect_count in range(11):
            request_kwargs: dict = {
                "headers": current_headers,
                "timeout": timeout,
                "follow_redirects": False,
            }
            if current_params is not None:
                request_kwargs["params"] = current_params
            if current_json is not None:
                request_kwargs["json"] = current_json

            async with client.stream(current_method, current_url, **request_kwargs) as resp:
                status = resp.status_code
                if status in (301, 302, 303, 307, 308):
                    location = resp.headers.get("location")
                    if not follow_redirects or not location or redirect_count == 10:
                        return status, b""
                    next_url = urljoin(current_url, location)
                    if not await _url_is_allowed(next_url, allowed_hosts):
                        logger.warning("Bounded request rejected an unapproved redirect host")
                        return status, b""
                    next_origin = _origin(next_url)
                    if next_origin != _origin(current_url):
                        current_headers = _strip_secret_headers(current_headers)
                        next_url = _strip_secret_query(next_url)
                    current_url = next_url
                    current_params = None
                    if status == 303 or (status in (301, 302) and current_method not in ("GET", "HEAD")):
                        current_method = "GET"
                        current_json = None
                    continue

                if status not in accepted_statuses:
                    return status, b""

                content_length = resp.headers.get("content-length")
                if content_length:
                    try:
                        if int(content_length) > max_bytes:
                            logger.warning(
                                "Upstream response exceeds byte limit (%s > %s) at %s",
                                content_length, max_bytes, safe_url_for_logging(current_url),
                            )
                            return status, b""
                    except ValueError:
                        pass

                chunks: list[bytes] = []
                total = 0
                async for chunk in resp.aiter_bytes(chunk_size=64 * 1024):
                    total += len(chunk)
                    if total > max_bytes:
                        logger.warning(
                            "Upstream streamed response exceeds byte limit at %s",
                            safe_url_for_logging(current_url),
                        )
                        return status, b""
                    chunks.append(chunk)
                return status, b"".join(chunks)
        return 0, b""
    except httpx.TimeoutException:
        logger.warning("Bounded request timed out for %s", safe_url_for_logging(url))
        return 0, b""
    except Exception as exc:
        logger.warning(
            "Bounded request failed for %s: %s",
            safe_url_for_logging(url), type(exc).__name__,
        )
        return 0, b""


async def bounded_fetch_json(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    params: dict | None = None,
    json_body: dict | None = None,
    timeout: float | None = None,
    max_bytes: int = MAX_HTML_RESPONSE_BYTES,
    accepted_statuses: tuple[int, ...] = (200,),
    follow_redirects: bool = False,
    allowed_hosts: set[str] | frozenset[str] | None = None,
) -> tuple[int, object | None]:
    """Fetch bounded JSON metadata without materializing an unrestricted body."""
    import json

    status, body = await bounded_request_bytes(
        client,
        method,
        url,
        headers=headers,
        params=params,
        json_body=json_body,
        timeout=timeout,
        max_bytes=max_bytes,
        accepted_statuses=accepted_statuses,
        follow_redirects=follow_redirects,
        allowed_hosts=allowed_hosts,
    )
    if not body:
        return status, None
    try:
        return status, json.loads(body)
    except (UnicodeDecodeError, ValueError):
        return status, None


async def bounded_download_bytes(
    client: httpx.AsyncClient,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    params: dict | None = None,
    timeout: float | None = None,
    max_bytes: int = MAX_UPSTREAM_DOWNLOAD_BYTES,
    follow_redirects: bool = True,
    allowed_hosts: set[str] | frozenset[str] | None = None,
) -> tuple[int, bytes]:
    """Stream download with hard byte limit.

    Performs one streaming GET per redirect hop (at most eleven hops).
    Returns (status_code, data). On non-200, returns (status_code, b"") without
    reading the response body. On success, returns (200, data).

    Returns an empty body if a size limit is exceeded. Redirects are followed
    explicitly so provider secrets are removed before cross-origin requests.
    """
    return await bounded_request_bytes(
        client,
        "GET",
        url,
        headers=headers,
        params=params,
        timeout=timeout,
        max_bytes=max_bytes,
        follow_redirects=follow_redirects,
        allowed_hosts=allowed_hosts,
    )


async def bounded_fetch_text(
    client: httpx.AsyncClient,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    params: dict | None = None,
    timeout: float | None = None,
    max_bytes: int = MAX_HTML_RESPONSE_BYTES,
    follow_redirects: bool = True,
    allowed_hosts: set[str] | frozenset[str] | None = None,
) -> tuple[int, str]:
    """Bounded HTML/text fetch with size limit.

    Performs exactly ONE streaming GET request.
    Returns (status_code, text). On non-200, returns (status_code, "") without
    reading the response body. On success, returns (200, text).

    Returns (0, "") on timeout/error.
    """
    status, data = await bounded_request_bytes(
        client,
        "GET",
        url,
        headers=headers,
        params=params,
        timeout=timeout,
        max_bytes=max_bytes,
        accepted_statuses=(200,),
        follow_redirects=follow_redirects,
        allowed_hosts=allowed_hosts,
    )
    return status, data.decode("utf-8", errors="replace") if data else ""


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
