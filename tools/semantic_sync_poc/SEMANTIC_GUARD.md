# Experimental semantic guard

Run from the repository root using the existing environment:

```sh
HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/semantic_sync_poc/semantic_guard.py \
  --arabic path/to/original.srt --reference path/to/reference.srt \
  --alass path/to/alass.srt --output tools/semantic_sync_poc/result.srt
```

Uses the cached multilingual MiniLM model from the existing experiments (CPU),
numpy and sentence-transformers. No production imports or integration.
Outputs SRT and an adjacent `.srt.json` decision report. Optional `--model` and
`--report` override the model and report destination. Existing outputs are never
overwritten; choose fresh paths. Inputs must be UTF-8 SRT; BOM, CRLF, nonnumeric
cue labels, repeated blank separators, decimal timestamps and common arrow
variants are accepted. Invalid timing and mismatched Arabic/Alass text or order
are rejected before encoding. Labels are renumbered; dialogue and markup remain.

Add `--observe-only` to run the experimental monitoring path: all Alass timings
are retained, `corrected` stays empty, and proposed repairs are listed separately
in `proposed_corrections`. This is an offline CLI mode, not a production service.

Matching follows the earlier experiment: mutual nearest neighbours, cosine
similarity >= .75, margins >= .15 in both languages, maximum-weight monotonic
chain. Empty markup-only text is excluded without dropping cue indices.
A deterministic median pairwise slope initializes a MAD-trimmed linear fit.
Five-fold interleaved cross-validation fits timing inliers on training folds
only. No film names, cue IDs or timing parameters are embedded in the algorithm.

Confidence requires >=20 timing inliers, >=80% inlier retention, >=60% timeline
coverage, held-out median error <=.5s and P90 <=1s. All cues are scanned against
the fitted model with threshold max(5s, 4*CV P90). Corrections require direct
semantic support, <=1s held-out error, a reference-confirmed Alass start error,
and interpolation inside the inlier range. Unsupported and end-only differences
are reported for review, not repaired. Both endpoints use the fitted model for
an accepted repair; negative starts are clamped to zero, and collapsed intervals
are left untouched. Low confidence produces an explicitly reported Alass
pass-through. Exit 0 means output was produced, not necessarily confidence;
inspect `confident`, `reason`, `corrected` and `suspicious` in JSON.

This conservative prototype does not solve cut/scene differences or validate
against audio. Semantic anchors are pseudo-ground truth; timing CV is not an
independent human-labelled accuracy benchmark. It can leave real errors in
unsupported cues. Matrix matching uses O(Arabic cues * reference cues) memory.

## Validation

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/semantic_sync_poc/test_semantic_guard.py
.venv/bin/ruff check tools/semantic_sync_poc/semantic_guard.py tools/semantic_sync_poc/test_semantic_guard.py
```

Smoke inputs:

| Film | Arabic | Reference | Alass |
|---|---|---|---|
| The Collection | `subs_cache/43d229491b6048d2.srt` | `subs_cache/references/tt1748227_8ae53ebaa27a07affa2e_subdl_edition.srt` | `/tmp/the-collection.alass-sync.srt` |
| A Better Tomorrow | `subs_cache/de8099e661d23434.srt` | `subs_cache/references/tt0092263_52346554f475b29d48e9_subdl_team.srt` | `/tmp/better-tomorrow.alass.srt` |

Generated SRTs, JSON reports and smoke logs remain in this experimental directory.


## Checkpoint scope

The checkpoint retains the Python tools and Markdown validation summaries.
Generated validation directories, SRTs, per-SRT JSON reports, logs, model files
and caches are local artifacts excluded by the experiment-only `.gitignore`.
Paths to generated artifacts in the validation summary describe the original
local run; they are not included in a fresh checkout. Recreate them with the
documented commands and the separately available subtitle inputs/model cache.
The existing `--observe-only` CLI option is part of this PoC checkpoint; the next
Observation Mode stage has not been implemented or integrated into AutoSync.
