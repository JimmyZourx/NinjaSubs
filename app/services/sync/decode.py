"""Secure decoding adapter for externally downloaded reference subtitles.

All ZIP parsing and extraction is delegated to the application's guarded
extractor. This module intentionally contains no ZIP reader of its own.
"""

from __future__ import annotations

from app.extractor import (
    MAX_SUBTITLE_ENTRY_BYTES,
    MAX_ZIP_INPUT_BYTES,
    SubtitleExtractionError,
    extract_srt_from_zip,
    is_ass_subtitle,
    is_vtt_subtitle,
    transcode_to_utf8,
)


def decode_payload(
    raw: bytes,
    release_name: str,
    season: int | None = None,
    episode: int | None = None,
    min_bytes: int = 5120,
) -> bytes | None:
    """Decode one bounded archive/plain subtitle or return ``None`` safely."""
    if not raw or len(raw) > MAX_ZIP_INPUT_BYTES:
        return None
    try:
        if raw.startswith(b"PK"):
            decoded = extract_srt_from_zip(
                raw,
                target_filename=release_name,
                season=season,
                episode=episode,
            )
        else:
            if len(raw) > MAX_SUBTITLE_ENTRY_BYTES:
                return None
            decoded = transcode_to_utf8(raw)
    except SubtitleExtractionError:
        return None
    if len(decoded) <= min_bytes or not (
        b"-->" in decoded or is_ass_subtitle(decoded) or is_vtt_subtitle(decoded)
    ):
        return None
    return decoded
