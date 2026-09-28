"""Reference lookup parameters shared by AutoSync strategies."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field


@dataclass
class ReferenceQuery:
    """Media and edition context used to locate a trusted reference subtitle."""

    imdb_id: str
    target_filename: str | None = None
    release_group: str | None = None
    media_type: str = "movie"
    title: str | None = None
    year: int | None = None
    video_hash: str | None = None
    video_size: int | str | None = None
    stream_url: str | None = field(default=None, repr=False)
    season: int | None = None
    episode: int | None = None
    api_keys: dict[str, str] = field(default_factory=dict, repr=False)
    languages: tuple[str, ...] = ("eng",)

    @property
    def is_series(self) -> bool:
        return self.media_type.lower() in {"series", "tv", "anime"} or self.season is not None

    @property
    def effective_group(self) -> str:
        from app.services.sync.matching import _release_group

        group = self.release_group or _release_group(self.target_filename) or "unknown"
        return re.sub(r"[^A-Za-z0-9]+", "", group) or "unknown"

    @property
    def cache_stem(self) -> str:
        """Versioned media fingerprint derived only from stable, non-secret inputs."""
        imdb = re.sub(r"[^A-Za-z0-9]+", "", self.imdb_id or "unknown") or "unknown"
        identity = json.dumps(
            [
                self.media_type.lower(),
                re.sub(r"\s+", " ", (self.target_filename or "").strip().lower()),
                (self.video_hash or "").strip().lower(),
                str(self.video_size or ""),
                self.languages,
                self.season,
                self.episode,
            ],
            ensure_ascii=False,
        )
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
        return f"{imdb}_{digest}"


@dataclass(frozen=True)
class ResolvedReference:
    """Reference subtitle content and the evidence supporting its selection."""

    text: str | None
    kind: str = "abort"
    bluray_match: bool = False
    candidate: str = ""
