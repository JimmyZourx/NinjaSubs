"""Regression matrix pinning current release-name normalization behavior.

Phase 1 of canonical release-name normalization (Approach B).
These tests pin CURRENT behavior before any production refactor.
"""

import sys

from app.models import SubtitleRelease
from app.services.aggregator import clean_subtitle_display_name
from app.services.ranking import sanitize_release_name
from app.services.subtitle_matcher import deduplicate_subtitles
from app.services.sync.matching import (
    _codec_kind,
    _edition_tags,
    _release_group,
    _source_kind,
)

# --- 6.1 Ranking: sanitize_release_name (must not change) ---


def test_ranking_strips_video_extensions():
    assert sanitize_release_name("Movie.2024.1080p.BluRay.x265-GROUP.srt") == (
        "Movie.2024.1080p.BluRay.x265-GROUP"
    )
    assert sanitize_release_name("Show.S01E01.WEB-DL.NF.1080p.x264-NTb.mkv") == (
        "Show.S01E01.WEB-DL.NF.1080p.x264-NTb"
    )


def test_ranking_strips_metadata_brackets_keeps_group_brackets():
    assert sanitize_release_name("[Group] Movie 2024 [1080p]") == "[Group] Movie 2024"


def test_ranking_empty_and_falsy_input():
    assert sanitize_release_name("") == ""
    assert sanitize_release_name(None) == ""


# --- 6.6 Web variants: ranking must preserve ---


def test_ranking_preserves_web_variants():
    assert sanitize_release_name("Movie.2024.WEB-DL.AMAZN.1080p.x265") == (
        "Movie.2024.WEB-DL.AMAZN.1080p.x265"
    )
    assert sanitize_release_name("Movie.2024.WEBRip.DSNP.720p.x264") == (
        "Movie.2024.WEBRip.DSNP.720p.x264"
    )
    assert sanitize_release_name("Show.S01E01.WEB-DL.NF.2160p.HDR.x265") == (
        "Show.S01E01.WEB-DL.NF.2160p.HDR.x265"
    )


# --- 6.5 Anime fansubs: ranking must preserve ---


def test_ranking_anime_fansub_collapses_duplicate_whitespace():
    # DESIRED: (1080p) parens removed, duplicate whitespace collapsed to single space.
    # [ABCD1234] CRC bracket preserved for ranking.
    assert sanitize_release_name(
        "[SubsPlease] One Piece - 1085 (1080p) [ABCD1234].mkv"
    ) == "[SubsPlease] One Piece - 1085 [ABCD1234]"


def test_ranking_anime_version_tag_preserved():
    assert sanitize_release_name("[Group] Show - 12 [v2]") == "[Group] Show - 12 [v2]"


# --- 6.7 Arabic/Latin mixed: ranking must preserve ---


def test_ranking_arabic_latin_mixed_preserved():
    assert sanitize_release_name("فيلم-2024-1080p-WEB-DL.x264") == (
        "فيلم-2024-1080p-WEB-DL.x264"
    )
    assert sanitize_release_name("[عرب سب] فيلم 2024") == "[عرب سب] فيلم 2024"


# --- Hash handling in ranking (pinned current) ---


def test_ranking_strips_hex_hash_suffix():
    assert (
        sanitize_release_name("Movie.2024.1080p.BluRay.x265-GROUP_86f1f22e8f1fd5bd")
        == "Movie.2024.1080p.BluRay.x265-GROUP"
    )


def test_ranking_collapses_duplicate_whitespace():
    # DESIRED (secondary): token removal must not leave duplicate spaces.
    assert sanitize_release_name("Movie  Name   2024") == "Movie Name 2024"


def test_ranking_strips_non_hex_generated_suffix():
    # DESIRED: generated non-hex suffix _hash123 is stripped like hex hashes.
    assert sanitize_release_name("Movie.2024.1080p.BluRay.x265-GROUP_hash123") == (
        "Movie.2024.1080p.BluRay.x265-GROUP"
    )


# --- 6.2 Dedup: dedup key stability (pinned current) ---


def test_dedup_key_extension_variant_same():
    a = sanitize_release_name("Movie.2024.1080p.BluRay.x265-GROUP.srt").lower()
    b = sanitize_release_name("Movie.2024.1080p.BluRay.x265-GROUP").lower()
    assert a == b


def test_dedup_key_hex_hash_variant_same():
    a = sanitize_release_name("Movie.2024.1080p.BluRay.x265-GROUP_86f1f22e8f1fd5bd").lower()
    b = sanitize_release_name("Movie.2024.1080p.BluRay.x265-GROUP").lower()
    assert a == b


def test_dedup_key_non_hex_hash123_variant_equivalent():
    # DESIRED: generated suffix stripped -> equivalent dedup keys.
    a = sanitize_release_name("Movie.2024.1080p.BluRay.x265-GROUP_hash123").lower()
    b = sanitize_release_name("Movie.2024.1080p.BluRay.x265-GROUP").lower()
    assert a == b


def test_deduplicate_subtitles_hash123_pair_merged():
    # DESIRED: equivalent releases with generated suffix merge to one candidate.
    subs = [
        SubtitleRelease(
            release_name="Movie.2024.1080p.BluRay.x265-GROUP_hash123",
            download_url="http://x/1",
            provider="subdl",
            lang="ara",
        ),
        SubtitleRelease(
            release_name="Movie.2024.1080p.BluRay.x265-GROUP",
            download_url="http://x/2",
            provider="subsource",
            lang="ara",
        ),
    ]
    out = deduplicate_subtitles(subs)
    assert len(out) == 1


def test_deduplicate_subtitles_hex_hash_pair_merged():
    subs = [
        SubtitleRelease(
            release_name="Movie.2024.1080p.BluRay.x265-GROUP_86f1f22e8f1fd5bd",
            download_url="http://x/1",
            provider="subdl",
            lang="ara",
        ),
        SubtitleRelease(
            release_name="Movie.2024.1080p.BluRay.x265-GROUP",
            download_url="http://x/2",
            provider="subsource",
            lang="ara",
        ),
    ]
    out = deduplicate_subtitles(subs)
    assert len(out) == 1


# --- 6.3 AutoSync: group/source/edition detection (must not change) ---


def test_autosync_group_source_codec_resolution_webdl():
    name = "Movie.2024.1080p.WEB-DL.NF.x264-NTb"
    assert _release_group(name) == "NTb"
    assert _source_kind(name) == "webdl"
    assert _codec_kind(name) == "h264"
    assert _source_kind(name) is not None
    assert _release_group(name) is not None


def test_autosync_group_source_bluray_remux():
    name = "Movie.2024.1080p.BluRay.REMUX.x265-SPARKS"
    assert _release_group(name) == "SPARKS"
    assert _source_kind(name) == "remux"
    assert _codec_kind(name) == "h265"


def test_autosync_bracketed_fansub_group_none_pinned():
    # PINNED CURRENT: bracketed [E-Subs] yields group None (spec expected 'E-Subs').
    name = "[E-Subs] Show S01E01"
    assert _release_group(name) is None
    assert _source_kind(name) is None


def test_autosync_anime_crc_trailing_bracket_group():
    name = "[SubsPlease] One Piece - 1085 (1080p) [ABCD1234]"
    assert _release_group(name) == "ABCD1234"
    assert _source_kind(name) is None
    assert _resolution_of(name) == "1080p"


def _resolution_of(name: str):
    from app.services.sync.matching import _resolution

    return _resolution(name)


def test_autosync_arabic_group_preserved():
    name = "فيلم-2024-1080p-WEB-DL.x264-GRP"
    assert _release_group(name) == "GRP"
    assert _source_kind(name) == "webdl"
    assert _codec_kind(name) == "h264"


def test_autosync_arabic_bracket_group_none_pinned():
    name = "[عرب سب] فيلم 2024"
    assert _release_group(name) is None
    assert _source_kind(name) is None


def test_autosync_edition_tags_extended():
    name = "Movie.2024.Extended.1080p.BluRay.x265-GROUP"
    assert _release_group(name) == "GROUP"
    assert _source_kind(name) == "bluray"
    assert "extended" in _edition_tags(name)


def test_autosync_service_tag_nf_detection():
    name = "Show.S01E01.WEB-DL.NF.2160p.HDR.x265-NTb"
    assert _release_group(name) == "NTb"
    assert _source_kind(name) == "webdl"
    assert _codec_kind(name) == "h265"
    assert _resolution_of(name) == "2160p"


# --- 6.4 Display: clean_subtitle_display_name (pinned current) ---


def test_display_strips_hex_hash_and_extension():
    assert clean_subtitle_display_name("Movie.srt_6e4b079e36dd0457") == "Movie"


def test_display_strips_leading_fansub_bracket():
    assert clean_subtitle_display_name("[MSRT Fansub] Show S01E01.srt") == "Show S01E01"


def test_display_preserves_parens_year():
    assert clean_subtitle_display_name("Movie (2024).mkv") == "Movie (2024)"


def test_display_keeps_technical_tokens():
    assert (
        clean_subtitle_display_name("Show.S01E01.WEB-DL.NF.1080p.x264-NTb.mkv")
        == "Show.S01E01.WEB-DL.NF.1080p.x264-NTb"
    )


def test_display_anime_fansub_pinned():
    # PINNED CURRENT: leading [SubsPlease] stripped, CRC [ABCD1234] kept, (1080p) kept.
    assert (
        clean_subtitle_display_name("[SubsPlease] One Piece - 1085 (1080p) [ABCD1234].mkv")
        == "One Piece - 1085 (1080p) [ABCD1234]"
    )


def test_display_bracketed_version_tag_leading_group_stripped():
    assert clean_subtitle_display_name("[Group] Show - 12 [v2]") == "Show - 12 [v2]"


def test_display_arabic_latin_mixed():
    assert clean_subtitle_display_name("فيلم-2024-1080p-WEB-DL.x264") == (
        "فيلم-2024-1080p-WEB-DL.x264"
    )


def test_display_arabic_bracket_group_stripped():
    assert clean_subtitle_display_name("[عرب سب] فيلم 2024") == "فيلم 2024"


def test_display_non_hex_hash123_suffix_not_stripped_pinned():
    # PINNED CURRENT: non-hex suffix survives display cleanup too.
    assert clean_subtitle_display_name("Movie.2024.1080p.BluRay.x265-GROUP_hash123") == (
        "Movie.2024.1080p.BluRay.x265-GROUP_hash123"
    )


if __name__ == "__main__":
    sys.exit(0)
