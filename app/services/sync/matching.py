"""Pure release-name helpers used by AutoSync reference selection."""

from __future__ import annotations

import re
from urllib.parse import unquote_plus

_INFO_TOKENS = ("info", "nfo", "readme", "sample", "trailer", "credit", "cover")
_SEASON_PATTERNS = (
    re.compile(r"(?i)\bS(\d{1,2})[\s._-]?E\d{1,3}\b"),
    re.compile(r"(?i)\bS(\d{1,2})\b"),
    re.compile(r"(?i)\bSeason[\s._-]?(\d{1,2})\b"),
    re.compile(r"(?i)\b(\d{1,2})x\d{1,3}\b"),
)
_SEASON_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
    "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12,
}
_EPISODE_PATTERNS = (
    re.compile(r"(?i)\bS\d{1,2}[\s._-]?E(\d{1,3})\b"),
    re.compile(r"(?i)\bE(?:P)?[\s._-]?(\d{1,3})\b"),
    re.compile(r"(?i)\bEpisode[\s._-]?(\d{1,3})\b"),
    re.compile(r"(?i)\b\d{1,2}x(\d{1,3})\b"),
)
_RELEASE_TOKEN_REGEX = re.compile(
    r"(1080p|720p|2160p|480p|4k|web[\s._-]?dl|webrip|blu[\s._-]?ray|brrip|"
    r"bdrip|hdtv|dvdrip|x264|x265|h\.?264|h\.?265|hevc|aac|dts|proper|repack|"
    r"remux|s\d{1,2}[\s._-]?e\d{1,2}|season[\s._-]?\d+|19\d{2}|20\d{2})",
    re.IGNORECASE,
)
_SOURCE_PATTERNS = (
    ("remux", r"\b(bdremux|remux)\b"),
    ("bluray", r"\b(blu[\s._-]?ray|bdrip|brrip|bd25|bd50)\b"),
    ("webdl", r"\b(web[\s._-]?dl|webrip|web)\b"),
    ("stream", r"\b(nf|netflix|amzn|amazon|dsnp|disney|hmax|max|atvp|appletv|itunes|hulu)\b"),
    ("hdtv", r"\b(hdtv|pdtv|dsr|satrip)\b"),
    ("dvd", r"\b(dvdrip|dvd|dvdr)\b"),
    ("cam", r"\b(cam|ts|tc|hdts|hdcam)\b"),
)
_CODEC_PATTERNS = (
    ("h265", r"\b(x265|h\.?265|hevc)\b"),
    ("h264", r"\b(x264|h\.?264|avc)\b"),
    ("xvid", r"\b(xvid|divx)\b"),
    ("av1", r"\bav1\b"),
)
_EDITION_TAGS = frozenset(
    {"us", "uk", "intl", "international", "unrated", "rated", "extended", "directors",
     "theatrical", "uncut", "proper", "repack", "rerip", "imax", "remastered", "criterion"}
)
_RETAIL_DISC_SOURCES = frozenset({"bluray", "remux"})


def _season_number(name: str | None) -> int | None:
    base = (name or "").replace("\\", "/").rsplit("/", 1)[-1]
    for pattern in _SEASON_PATTERNS:
        match = pattern.search(base)
        if match:
            return int(match.group(1))
    match = re.search(r"(?i)\bseason[\s._-]?([a-z]+)\b", base)
    if match:
        return _SEASON_WORDS.get(match.group(1).lower())
    return None


def _member_episode_number(name: str) -> int | None:
    base = name.replace("\\", "/").rsplit("/", 1)[-1]
    for pattern in _EPISODE_PATTERNS:
        match = pattern.search(base)
        if match:
            return int(match.group(1))
    # Anime/season-pack filenames frequently use a bare episode number.
    match = re.search(r"(?i)(?:^|[._\-\s\[#])0*(\d{1,3})(?:$|[._\-\s\]#])", base)
    if match and int(match.group(1)) not in {0, 24, 25, 30, 50, 60, 264, 265, 480, 576, 720, 1080, 2160}:
        return int(match.group(1))
    return None


def _is_info_member(name: str) -> bool:
    lowered = name.replace("\\", "/").rsplit("/", 1)[-1].lower()
    return any(token in lowered for token in _INFO_TOKENS)


def _source_kind(name: str | None) -> str | None:
    lowered = (name or "").lower()
    for kind, pattern in _SOURCE_PATTERNS:
        if re.search(pattern, lowered):
            return kind
    return None


def _codec_kind(name: str | None) -> str | None:
    lowered = (name or "").lower()
    for kind, pattern in _CODEC_PATTERNS:
        if re.search(pattern, lowered):
            return kind
    return None


def _resolution(name: str | None) -> str | None:
    match = re.search(r"\b(2160p|1080p|720p|480p|4k)\b", name or "", re.IGNORECASE)
    return match.group(1).lower() if match else None


def _sources_compatible(target_source: str | None, candidate_source: str | None) -> bool:
    if not target_source or not candidate_source:
        return True
    return candidate_source == target_source or {target_source, candidate_source} <= _RETAIL_DISC_SOURCES


def is_retail_disc_source(kind: str | None) -> bool:
    return kind in _RETAIL_DISC_SOURCES


def _edition_tags(name: str | None) -> frozenset[str]:
    normalized = re.sub(r"[^a-z0-9]+", " ", (name or "").lower())
    tags = {token for token in normalized.split() if token in _EDITION_TAGS}
    for phrase in ("final cut", "special edition", "open matte"):
        if re.search(r"\b" + phrase.replace(" ", r"\s+") + r"\b", normalized):
            tags.add(phrase.replace(" ", "_"))
    return frozenset(tags)


def _release_group(name: str | None) -> str | None:
    if not name:
        return None
    base = re.sub(r"\.(srt|ass|ssa|sub|vtt|mkv|mp4|avi)$", "", name.rsplit("/", 1)[-1].strip(), flags=re.I)
    bracket = re.search(r"\(([^()]*)\)\s*$", base) or re.search(r"\[([^\[\]]*)\]\s*$", base)
    tokens = re.findall(r"[A-Za-z0-9]+", bracket.group(1)) if bracket else []
    if not tokens:
        dash = re.search(r"-\s*([^\-\s]+)\s*$", base)
        tokens = re.findall(r"[A-Za-z0-9]+", dash.group(1)) if dash else []
    ignored = {"1080p", "720p", "2160p", "480p", "4k", "web", "dl", "bluray", "brrip", "bdrip", "hdtv", "x264", "x265", "h264", "h265", "hevc", "aac", "dts", "remux", "srt", "ass", "ssa", "vtt", "eng", "en", "ar"}
    while tokens and (tokens[-1].lower() in ignored or tokens[-1].isdigit()):
        tokens.pop()
    return tokens[-1] if tokens and re.fullmatch(r"[A-Za-z0-9]{2,}", tokens[-1]) else None


def is_informative_release_name(name: str | None) -> bool:
    return bool(_RELEASE_TOKEN_REGEX.search((name or "").strip()))


def looks_like_season_pack(name: str | None) -> bool:
    lowered = (name or "").lower()
    return "complete" in lowered or bool(re.search(r"\bs\d{1,2}\s*-\s*s?\d{1,2}\b", lowered)) or (
        "season" in lowered and _member_episode_number(lowered) is None
    )


def prefer_meaningful_release_name(value: str | None) -> str:
    segments = [segment for segment in unquote_plus(str(value or "")).replace("\\", "/").split("/") if segment]
    if not segments:
        return str(value or "").strip()
    if is_informative_release_name(segments[-1]):
        return segments[-1]
    return next((segment for segment in reversed(segments[:-1]) if is_informative_release_name(segment)), segments[-1])


def _title_from_filename(filename: str | None) -> str | None:
    if not filename:
        return None
    base = unquote_plus(filename).replace("\\", "/").rsplit("/", 1)[-1]
    base = re.sub(r"\.[A-Za-z0-9]{2,5}$", "", base)
    marker = re.search(r"(?i)(?:^|[\s._-])(?:s\d{1,2}[\s._-]*e\d{1,3}|season[\s._-]*\d{1,2}|\d{1,2}x\d{1,3})", base)
    title = base[:marker.start()] if marker else base
    title = re.sub(r"[._\-+]+", " ", title)
    title = re.sub(r"\b(19\d{2}|20\d{2})\b", " ", title)
    title = re.sub(r"\s+", " ", title).strip(" -")
    return title if len(title) >= 3 else None
