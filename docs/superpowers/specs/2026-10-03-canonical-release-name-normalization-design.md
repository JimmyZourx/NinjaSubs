# Canonical Release-Name Normalization Design

**Date:** 2026-10-03
**Status:** Revised after Phase 1 evidence — narrowed scope
**Branch:** `feature/normalize-release-names`

---

## 1. Shared Understanding

**Intent:** Fix the smallest proven normalization bug: equivalent releases produce different normalized names and dedup keys.

**Priority ranking (preserved):**
1. Ranking accuracy
2. AutoSync matching
3. Dedup stability
4. Display consistency

**Constraint:** AutoSync matching semantics unchanged in this phase. No broad normalization abstraction unless failing tests prove it necessary.

---

## 2. Phase 1 Observed Behavior (evidence)

Regression matrix `tests/test_release_name_normalization.py` (31 tests) pins current behavior. Key observations:

### 2.1 Primary bug (proven)

`sanitize_release_name` strips trailing **hex** hashes (`_TRAILING_HASH = (?:_|\.)[a-f0-9]{8,}`) but not non-hex generated suffixes like `_hash123` (`h`, `s` are not hex digits).

```
sanitize_release_name("Movie.2024.1080p.BluRay.x265-GROUP_hash123")
  == "Movie.2024.1080p.BluRay.x265-GROUP_hash123"   # NOT stripped (bug)

sanitize_release_name("Movie.2024.1080p.BluRay.x265-GROUP_86f1f22e8f1fd5bd")
  == "Movie.2024.1080p.BluRay.x265-GROUP"            # hex stripped (correct)
```

**Dedup impact (proven):**
```python
deduplicate_subtitles([sub_hash123, sub_plain])  # → 2 kept (should merge)
deduplicate_subtitles([sub_hex,    sub_plain])  # → 1 kept (correct)
```

Dedup keys derive from `sanitize_release_name(release_name).lower()`; the unstripped suffix makes equivalent releases produce different keys.

### 2.2 Secondary observation (whitespace)

Ranking normalization can leave duplicate spaces:
```
sanitize_release_name("[SubsPlease] One Piece - 1085 (1080p) [ABCD1234].mkv")
  == "[SubsPlease] One Piece - 1085  [ABCD1234]"   # double space: (1080p) removed
```

### 2.3 AutoSync (pinned, NOT in scope)

`app/services/sync/matching.py` extraction pinned as-is:
- `_release_group("[E-Subs] Show S01E01")` → `None` (spec §6.3 had wrongly predicted `E-Subs`)
- `_release_group("[SubsPlease] ... [ABCD1234]")` → `"ABCD1234"` (trailing bracket, not leading)
- Arabic group names and service tags (NF/AMZN/DSNP) detected correctly — no changes needed.

### 2.4 Display (pinned, unchanged in this phase)

`clean_subtitle_display_name` shares the hex-only hash regex gap but display is priority 4; **this phase changes ranking/dedup normalization only**. Display remains pinned as current behavior.

---

## 3. Revised Design: Minimal Targeted Fix

### 3.1 Scope (narrowed)

**In scope:**
1. Strip trailing non-hex generated suffixes (`_hash123`, similar) in `sanitize_release_name` — fixes dedup equivalence.
2. Collapse duplicate whitespace left by token removal in `sanitize_release_name`.

**Out of scope (explicitly deferred):**
- Broad `normalize_release_name_core` abstraction — NOT introduced; failing tests do not require it.
- AutoSync `sync/matching.py` — untouched.
- Display `clean_subtitle_display_name` — untouched this phase.
- Any change to scoring, metadata parsing, or tier logic.

### 3.2 Minimal production change

In `app/services/ranking.py`:

1. Extend trailing-hash stripping to also match a non-hex generated suffix: trailing `_` + alphanumeric token that follows a hex-like or generated-id pattern. Concretely: strip trailing `_[A-Za-z0-9]{8,}` **only when** it looks like a generated id (contains digits and letters mixed) — implemented as one additional regex alongside `_TRAILING_HASH`.

2. Add whitespace collapse: `re.sub(r" {2,}", " ", name)` (or equivalent single pass) near the existing separator-normalization step.

**Public API:** `sanitize_release_name(raw_name) -> str` signature unchanged. All 10 ranking callers, dedup, and re-exports unaffected.

### 3.3 Why this is minimal

- One function modified (`sanitize_release_name`), two small regex/sub additions.
- No new modules, no core abstraction, no signature changes.
- AutoSync path untouched → semantics provably unchanged.
- Display path untouched → display pins stay green.

---

## 4. Invariants (Behavioral Guarantees)

1. **Ranking group/edition tokens preserved** — fix only targets trailing generated suffixes + duplicate spaces.
2. **Dedup keys become equivalent for `_hash123` vs plain variant** (the fix goal).
3. **AutoSync detection unchanged** — `sync/matching.py` not modified.
4. **Web-variant, Arabic, anime tokens preserved** — all existing pins stay green.
5. **Public APIs stable** — no signature or export changes.

---

## 5. Test Plan (Phase 2, TDD)

### Failing tests first (before production change):

| Test | Expected (desired) | Current (failing) |
|------|-------------------|-------------------|
| `test_ranking_strips_non_hex_generated_suffix` | `_hash123` removed | suffix kept |
| `test_dedup_key_hash123_variant_equivalent` | keys equal | keys differ |
| `test_deduplicate_subtitles_hash123_pair_merged` | out length 1 | out length 2 |
| `test_ranking_collapses_duplicate_whitespace` (secondary) | single space | double space |

### Then smallest fix, then:
- Focused file `tests/test_release_name_normalization.py` — all green (old pins + new tests)
- Full suite — 875 tests green
- Ruff + mypy

---

## 6. Files Touched

**Modified:**
- `docs/superpowers/specs/2026-10-03-canonical-release-name-normalization-design.md` (this file, revision)
- `tests/test_release_name_normalization.py` — add failing tests for desired behavior
- `app/services/ranking.py` — minimal `sanitize_release_name` fix (Phase 2)

**Not modified:**
- `app/services/sync/*` — AutoSync untouched
- `app/services/aggregator.py` — display untouched
- `app/services/subtitle_matcher.py` — dedup logic untouched (uses ranking normalizer)
- `app/services/cache.py`, `app/utils/release_matcher.py` — untouched

---

## 7. Success Criteria

- [ ] Spec reflects actual Phase 1 observed behavior
- [ ] New tests fail for expected reason before fix
- [ ] Minimal fix makes new tests pass without breaking 31 existing pins
- [ ] Full suite (875) green
- [ ] AutoSync semantics unchanged (no `sync/` diff)
- [ ] Ruff + mypy pass
