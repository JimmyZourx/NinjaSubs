"""Auditable candidate decision tree: exact team, compatible edition, then abort."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from app.services.sync.matching import (
    _codec_kind,
    _edition_tags,
    _member_episode_number,
    _release_group,
    _resolution,
    _season_number,
    _source_kind,
    _sources_compatible,
    looks_like_season_pack,
)

if TYPE_CHECKING:
    from app.services.sync.query import ReferenceQuery

logger = logging.getLogger(__name__)
DecisionKind = Literal["team", "edition", "abort"]


@dataclass(frozen=True)
class ReferenceDecision:
    kind: DecisionKind
    release: Any | None
    reason: str


def _release_name(release: Any) -> str:
    return str(getattr(release, "release_name", "") or "")


def _edition_compatible(name: str, query: ReferenceQuery, strict: bool) -> bool:
    target = query.target_filename or ""
    target_tags, candidate_tags = _edition_tags(target), _edition_tags(name)
    if strict and target_tags != candidate_tags:
        return False
    target_source, candidate_source = _source_kind(target), _source_kind(name)
    if target_source and candidate_source and not _sources_compatible(target_source, candidate_source):
        return False
    if strict and (target_source is None or candidate_source is None):
        return False
    target_codec, candidate_codec = _codec_kind(target), _codec_kind(name)
    if target_codec and candidate_codec and target_codec != candidate_codec:
        return False
    if strict and not _sources_compatible(target_source, candidate_source):
        return False
    target_res, candidate_res = _resolution(target), _resolution(name)
    if target_res and candidate_res and target_res != candidate_res:
        if not (is_retail_disc_source(target_source) and {target_res, candidate_res} <= {"1080p", "720p"}):
            return False
    if strict and (target_res is None or candidate_res is None) and not is_retail_disc_source(target_source):
        return False
    return True


def is_retail_disc_source(kind: str | None) -> bool:
    return kind in {"bluray", "remux"}


def decide(
    candidates: list[Any],
    query: ReferenceQuery,
    *,
    strict: bool = True,
    provider_name: str = "?",
) -> ReferenceDecision:
    """Filter season/episode mismatches and choose only a defensible reference."""
    pool = list(candidates)
    if query.season is not None:
        season_matches = [item for item in pool if _season_number(_release_name(item)) == query.season]
        if not season_matches:
            return ReferenceDecision("abort", None, f"no candidate carries season {query.season}")
        pool = season_matches
    if query.episode is not None:
        pool = [
            item for item in pool
            if _member_episode_number(_release_name(item)) in (None, query.episode)
        ]
        if not pool:
            return ReferenceDecision("abort", None, "no candidate carries the requested episode")

    target_group = _release_group(query.target_filename)
    if target_group:
        team = [
            item for item in pool
            if (_release_group(_release_name(item)) or "").lower() == target_group.lower()
        ]
        team = [item for item in team if _edition_tags(_release_name(item)) == _edition_tags(query.target_filename)]
        team = [
            item for item in team
            if _sources_compatible(_source_kind(query.target_filename), _source_kind(_release_name(item)))
        ]
        if team:
            return ReferenceDecision("team", team[0], f"exact release group {target_group}")

    target_name = query.target_filename or ""
    has_edition_info = any(
        (_release_group(target_name), _source_kind(target_name), _codec_kind(target_name), _resolution(target_name))
    ) or bool(_edition_tags(target_name))
    if not has_edition_info:
        if strict:
            return ReferenceDecision("abort", None, "target carries no edition information")
        singles = [item for item in pool if not looks_like_season_pack(_release_name(item))]
        selected = (singles or pool)
        return ReferenceDecision("edition", selected[0], "relaxed fallback for uninformative target") if selected else ReferenceDecision("abort", None, "no candidate")

    edition = [item for item in pool if _edition_compatible(_release_name(item), query, strict)]
    if edition:
        target_res = _resolution(target_name)
        edition.sort(key=lambda item: _resolution(_release_name(item)) != target_res)
        return ReferenceDecision("edition", edition[0], "source/codec/edition match")
    logger.info("[reference] %s: no compatible reference candidate", provider_name)
    return ReferenceDecision("abort", None, "no candidate proves the same edition")
