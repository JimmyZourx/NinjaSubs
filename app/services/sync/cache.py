"""Persistent cache of validated reference subtitle text and verdict metadata."""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import re
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse

from app.config import settings
from app.extractor import MAX_SUBTITLE_ENTRY_BYTES
from app.services.sync.query import ReferenceQuery, ResolvedReference

logger = logging.getLogger(__name__)
_DEFAULT_TTL_SECONDS = 30 * 24 * 60 * 60
_KNOWN_KINDS = frozenset({"team", "edition", "hash", "embedded"})
_EXACT_KINDS = frozenset({"team", "hash", "embedded"})
_KNOWN_PROVIDERS = frozenset({"subdl", "subsource", "podnapisi", "opensubtitles"})
_MAX_SIDECAR_BYTES = 4096


class ReferenceDiskCache:
    """Disk cache keyed by a sanitized media fingerprint, never by credentials."""

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        ttl: float | None = None,
        min_bytes: int = 5120,
    ) -> None:
        configured = root or getattr(settings, "REFERENCE_CACHE_DIR", None)
        self.root = Path(configured) if configured else Path(settings.CACHE_DIR) / "references"
        if ttl is None:
            try:
                ttl = float(getattr(settings, "REFERENCE_CACHE_TTL_SECONDS", _DEFAULT_TTL_SECONDS))
            except (TypeError, ValueError):
                ttl = _DEFAULT_TTL_SECONDS
        self.ttl = max(0.0, ttl)
        self.min_bytes = max(0, min_bytes)

    @staticmethod
    def _safe_candidate(value: object) -> str:
        """Store bounded release provenance, never a URL or signed token."""
        candidate = re.sub(r"[\x00-\x1f\x7f]+", " ", str(value or ""))
        candidate = re.sub(r"\s+", " ", candidate).strip()[:256]
        parsed = urlparse(candidate)
        is_url = bool(parsed.netloc and (parsed.scheme or candidate.startswith("//")))
        has_signed_query = bool(
            re.search(
                r"(?i)(?:^|[?&;])(?:token|sig|signature|api[_-]?key|x-amz-signature)\s*=",
                candidate,
            )
        )
        if is_url or has_signed_query:
            return ""
        return candidate

    @staticmethod
    def _sidecar(path: Path) -> Path:
        return path.with_suffix(path.suffix + ".json")

    @staticmethod
    def _kind_from_name(path: Path) -> str:
        suffix = path.stem.rsplit("_", 1)[-1]
        return suffix if suffix in _KNOWN_KINDS else "edition"

    @staticmethod
    def _provider_from_name(path: Path, cache_stem: str) -> str | None:
        prefix = f"{cache_stem}_"
        if not path.stem.startswith(prefix):
            return None
        parts = path.stem[len(prefix):].rsplit("_", 1)
        if len(parts) != 2 or parts[0] not in _KNOWN_PROVIDERS:
            return None
        return parts[0]

    def _paths(self, query: ReferenceQuery) -> list[Path]:
        return sorted(self.root.glob(f"{query.cache_stem}_*.srt")) if self.root.is_dir() else []

    def get(self, query: ReferenceQuery) -> ResolvedReference | None:
        """Return the best unexpired, integrity-checked reference or a cache miss."""
        options: list[tuple[int, str, ResolvedReference]] = []
        for path in self._paths(query):
            try:
                stat = path.stat()
                if stat.st_size > MAX_SUBTITLE_ENTRY_BYTES:
                    continue
                sidecar = self._sidecar(path)
                if sidecar.stat().st_size > _MAX_SIDECAR_BYTES:
                    continue
                if time.time() - stat.st_mtime > self.ttl:
                    with contextlib.suppress(OSError):
                        path.unlink()
                        self._sidecar(path).unlink()
                    continue
                text = path.read_text(encoding="utf-8")
                verdict = json.loads(self._sidecar(path).read_text(encoding="utf-8"))
            except (OSError, UnicodeError, ValueError):
                continue
            kind = self._kind_from_name(path)
            if (
                self._provider_from_name(path, query.cache_stem) is None
                or not isinstance(verdict, dict)
                or verdict.get("kind") != kind
                or verdict.get("content_sha") != hashlib.sha256(text.encode()).hexdigest()
                or (settings.SYNC_REQUIRE_EXACT_MATCH and verdict.get("strict") is not True)
                or len(text.encode("utf-8")) <= self.min_bytes
                or "-->" not in text
            ):
                continue
            ref = ResolvedReference(
                text,
                kind,
                verdict.get("bluray_match") is True and kind != "abort",
                self._safe_candidate(verdict.get("candidate")),
            )
            options.append((0 if kind in _EXACT_KINDS else 1, path.name, ref))
        return min(options, key=lambda item: (item[0], item[1]))[2] if options else None

    def set(
        self,
        query: ReferenceQuery,
        source: str | None,
        text: str,
        kind: str = "edition",
        *,
        bluray_match: bool = False,
        candidate: str = "",
    ) -> None:
        """Persist a bounded reference and integrity/provenance sidecar atomically."""
        encoded = text.encode("utf-8") if isinstance(text, str) else b""
        if not encoded or len(encoded) > MAX_SUBTITLE_ENTRY_BYTES or b"-->" not in encoded:
            return
        provider = (source or "").strip().lower()
        if provider not in _KNOWN_PROVIDERS:
            return
        kind = kind if kind in _KNOWN_KINDS else "edition"
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            path = self.root / f"{query.cache_stem}_{provider}_{kind}.srt"
            meta = {
                "kind": kind,
                "bluray_match": bool(bluray_match),
                "candidate": self._safe_candidate(candidate),
                "strict": bool(settings.SYNC_REQUIRE_EXACT_MATCH),
                "content_sha": hashlib.sha256(encoded).hexdigest(),
            }
            self._atomic_write(path, text)
            self._atomic_write(self._sidecar(path), json.dumps(meta, ensure_ascii=False))
        except OSError as exc:
            logger.warning("[reference] cache write failed: %s", type(exc).__name__)

    def clear(self) -> None:
        """Clear reference cache files; caller is responsible for admin authorization."""
        if not self.root.is_dir():
            return
        for path in self.root.iterdir():
            if path.is_file():
                with contextlib.suppress(OSError):
                    path.unlink()

    @staticmethod
    def _atomic_write(path: Path, value: str) -> None:
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
            ) as handle:
                tmp_path = Path(handle.name)
                handle.write(value)
            tmp_path.replace(path)
        finally:
            if tmp_path is not None:
                with contextlib.suppress(OSError):
                    tmp_path.unlink()
