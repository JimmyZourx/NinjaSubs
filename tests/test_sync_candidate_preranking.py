"""TDD tests for hashless AutoSync reference pre-ranking using existing compatibility.

These tests are intentionally failing before the production change.
"""
from __future__ import annotations

from app.models import SubtitleRelease
from app.services.sync.external_strategy import _select_candidates_ranked
from app.services.sync.query import ReferenceQuery


def _make_release(release_name: str, provider="subdl", lang="en", is_hash_match=False, hearing_impaired=False):
    return SubtitleRelease(
        release_name=release_name,
        download_url=f"https://example.com/{release_name}",
        provider=provider,
        lang=lang,
        is_hash_match=is_hash_match,
        hearing_impaired=hearing_impaired,
    )


def test_same_tier_stronger_metadata_match_wins():
    """TEST 1 — same tier, stronger metadata match wins."""
    query = ReferenceQuery(
        imdb_id="tt1234567",
        target_filename="Show.S01E01.2160p.WEB-DL.DDP5.1.Atmos.H.265-GroupA",
        media_type="series",
        season=1,
        episode=1,
        languages=("en",),
    )
    # Both candidates have same source family but different release group => TIER_SOURCE_EDITION
    cand_a = _make_release("Show.S01E01.1080p.WEBRip.x264-GroupB")
    cand_b = _make_release("Show.S01E01.2160p.WEB-DL.H.265-GroupC")
    ranked = _select_candidates_ranked([cand_a, cand_b], query)
    # B has stronger source/resolution/codec match to target
    assert ranked[0].release_name == cand_b.release_name, "stronger compatibility should rank first"


def test_tier_hierarchy_dominates_compatibility():
    """TEST 2 — tier hierarchy dominates compatibility."""
    query = ReferenceQuery(
        imdb_id="tt1234567",
        target_filename="Show.S01E01.2160p.WEB-DL.H.265-GroupA",
        media_type="series",
        season=1,
        episode=1,
        languages=("en",),
    )
    # A is EXACT_GROUP tier, B is SOURCE_EDITION tier but better filename match
    cand_a = _make_release("Show.S01E01.480p.HDTV.x264-GroupA")
    cand_b = _make_release("Show.S01E01.2160p.WEB-DL.H.265-GroupB")
    ranked = _select_candidates_ranked([cand_a, cand_b], query)
    assert ranked[0].release_name == cand_a.release_name, "tier must dominate compatibility"


def test_exact_moviehash_remains_absolute():
    """TEST 3 — exact MovieHash remains absolute reference tier winner."""
    query = ReferenceQuery(
        imdb_id="tt1234567",
        target_filename="Show.S01E01.1080p.WEB-DL.H.265-GroupA",
        media_type="series",
        season=1,
        episode=1,
        languages=("en",),
    )
    hash_cand = _make_release("Show.S01E01.480p.HDTV.x264-GroupX", is_hash_match=True)
    strong_cand = _make_release("Show.S01E01.1080p.WEB-DL.H.265-GroupA")
    ranked = _select_candidates_ranked([strong_cand, hash_cand], query)
    assert ranked[0].is_hash_match is True, "hash candidate must rank first"


def test_no_target_filename_preserves_behavior():
    """TEST 4 — no target filename, preserve current deterministic behavior."""
    query = ReferenceQuery(
        imdb_id="tt1234567",
        target_filename=None,
        media_type="series",
        season=1,
        episode=1,
        languages=("en",),
    )
    cand_a = _make_release("Show.S01E01.1080p.WEB-DL.H.265-GroupA")
    cand_b = _make_release("Show.S01E01.1080p.WEB-DL.H.265-GroupB")
    ranked1 = _select_candidates_ranked([cand_a, cand_b], query)
    ranked2 = _select_candidates_ranked([cand_a, cand_b], query)
    # Order should be deterministic and unchanged by missing target
    assert [r.release_name for r in ranked1] == [r.release_name for r in ranked2]


def test_deterministic():
    """TEST 5 — deterministic ordering."""
    query = ReferenceQuery(
        imdb_id="tt1234567",
        target_filename="Show.S01E01.2160p.WEB-DL.H.265-GroupA",
        media_type="series",
        season=1,
        episode=1,
        languages=("en",),
    )
    cands = [
        _make_release("Show.S01E01.1080p.WEB-DL.H.265-GroupA"),
        _make_release("Show.S01E01.2160p.WEB-DL.H.265-GroupB"),
        _make_release("Show.S01E01.720p.HDTV.x264-GroupA"),
    ]
    r1 = [r.release_name for r in _select_candidates_ranked(cands, query)]
    r2 = [r.release_name for r in _select_candidates_ranked(cands[::-1], query)]
    assert r1 == r2, "ordering must be deterministic regardless of input order"


def test_retry_behavior_unchanged():
    """TEST 6 — retry behavior unchanged: ranking only changes trial order."""
    # This test ensures _select_candidates_ranked still returns a list
    # and does not filter candidates. The orchestrator retry loop remains
    # responsible for advancing past rejected candidates.
    query = ReferenceQuery(
        imdb_id="tt1234567",
        target_filename="Show.S01E01.2160p.WEB-DL.H.265-GroupA",
        media_type="series",
        season=1,
        episode=1,
        languages=("en",),
    )
    cands = [
        _make_release("Show.S01E01.1080p.WEB-DL.H.265-GroupA"),
        _make_release("Show.S01E01.2160p.WEB-DL.H.265-GroupB"),
    ]
    ranked = _select_candidates_ranked(cands, query)
    assert len(ranked) == 2, "ranking must not drop candidates"
