# Expanded semantic guard validation - 2026-09-29

Branch: `experiment/arabic-semantic-sync-poc`. No production code changed.

## Outcome

Four additional films, five input pairs (two Sunshine releases), were tested.
The original two films were also repeated, giving seven pairs at each of two
Alass split penalties: **14 real-file comparisons**. Guard thresholds were not
tuned after seeing results. The four additional films produced no automatic
repairs. Ip Man 3 failed confidence and was passed through unchanged.

The earlier Collection outlier depends on Alass configuration: **split penalty
0.5 exactly reproduces the old parsed Alass output**, including cue 267; the
default penalty 7 does not produce that outlier. The same exact reproduction
holds for the previous A Better Tomorrow output at penalty 0.5.

The prototype is suitable for further offline observation, not yet a claim of
production readiness or broad superiority over Alass. In particular, it leaves
untrusted cues unchanged even when they appear wrong.

## Results on the additional films

All errors are seconds, measured at semantic anchor starts. Semantic MAE is
five-fold timing CV. Alass/guard errors are descriptive scores against the same
semantic pseudo-labels, not independent human or audio ground truth.

| Film / release | Anchors | Semantic CV MAE | Alass MAE, penalty 7 | Alass MAE, penalty 0.5 | Guard confidence | Repairs at either penalty |
|---|---:|---:|---:|---:|---|---:|
| Sunshine BluRay | 285 | 0.027 | 0.051 | 0.023 | Passed | 0 |
| Sunshine DVD | 285 | 0.027 | 0.068 | 0.028 | Passed | 0 |
| Deuce Bigalow | 195 | 0.267 | 0.326 | 0.285 | Passed | 0 |
| Ip Man 3 | 98 | 0.460 | 0.455 | 0.455 | Failed; unchanged | 0 |
| tt0120591 / BluRay | 323 | 0.258 | 0.258 | 0.258 | Passed | 0 |

Sunshine DVD fitted slope **1.0428623**, versus **1.0000710** for BluRay,
covering substantial drift as well as nearly matched timing. The two releases
are one film, not two independent films. Ip Man 3 had only 78/98 timing inliers
(79.6%, below 80%) and CV P90 1.798s (above 1s); no thresholds were relaxed.

At penalty 0.5, Deuce Bigalow had 23 flagged disagreements and tt0120591 had
three. None had enough direct semantic support for automatic correction.
These are review candidates, not confirmed errors. At penalty 7 neither pair
had disagreements above the guard threshold. No scene-cut mismatch was
independently confirmed in this sample.

## Original-film regression and setting comparison

| Case | Alass anchor MAE | Guard anchor MAE | Largest anchor error before / after | Corrected blocks |
|---|---:|---:|---:|---|
| Collection, penalty 0.5 | 0.753 | 0.446 | 25.526 / 3.388 | 267 |
| Collection, penalty 7 | 0.488 | 0.488 | 3.388 / 3.388 | None |
| Better Tomorrow, penalty 0.5 | 0.150 | 0.150 | 3.002 / 3.002 | None |
| Better Tomorrow, penalty 7 | 0.153 | 0.153 | 2.377 / 2.377 | None |

Reference timings were identical between the old and newly canonicalized
Collection reference. The only textual changes were four trailing spaces.
Penalty 0.5 reproduced both old Alass files exactly as parsed, so the setting
comparison accounts for the observed timing difference. This does not prove
penalty 7 is universally better: Sunshine's anchor MAE was lower at 0.5.

## Review of the 14 retained Collection disagreements

This is a textual/context review against nearby English reference cues. It is
not audiovisual verification. None of these blocks was manually forced into
the algorithm. Evidence is saved in
`validation-20260929/legacy-alass-review/the-collection.json`.

| Block(s) | Review | Decision |
|---|---|---|
| 83, 84, 85 | Arabic requests for help fit dialogue near model timing; Alass places them near different action lines. Repeated generic phrases prevent unique matching. | Likely mismatch; retain for review |
| 86 | Combined move/name cue fits the model neighbourhood better, but segmentation differs. | Retain for review |
| 91 | Arabic and nearby reference wording are not a clean direct match. | Unresolved |
| 268 | Short name call has no separate reliable reference cue. | Unresolved |
| 309 | Help request has a plausible reference near model time, but is highly repetitive. | Likely mismatch; retain for review |
| 310 | Model neighbourhood contains rescue dialogue; translation is not one-to-one. | Unresolved |
| 350 | Reference text itself includes an embedded cue number/timestamp, suggesting a missing separator. | Reference-quality issue; do not auto-repair |
| 366 | Short expletive lacks a reliable local counterpart. | Unresolved |
| 402 | Model timing is near an equivalent expletive, but this phrase repeats. | Likely mismatch; retain for review |
| 403 | Model is near corresponding exclamation; Alass differs by about 108s. | Strong textual suspicion; retain for review |
| 404 | Model is near corresponding help dialogue; Arabic contains a typo. | Likely mismatch; retain for review |
| 426 | Film-title card, not ordinary dialogue. | Leave unchanged |

Cue 267 remains the supported repair: its direct English match asks for Arkin.
The first Better Tomorrow cue is the credit **W I N D Y**, not dialogue; its
negative model start and Alass zero clamp explain why this cue merits separate
treatment. It was left unchanged.

## Controls and integrity

- Eight offline unit tests pass, including observation-only proposals without
  timing edits, robust fitting, mismatches, low confidence and ambiguity.
- Each of seven pairs passed a synthetic correct-anchor-timing control with
  no edits. This control sets anchor timestamps to reference timestamps; it is
  not an independently verified full-film timing benchmark.
- Reusing aligned Alass as the source caused no new changes across seven pairs.
- Wrong-film reference (Sunshine Arabic versus Deuce Bigalow reference) produced
  only three anchors, failed confidence, and preserved Alass.
- Three malformed cached files were rejected and recorded. One Sunshine file
  contains archive bytes despite its `.srt` extension; the other failures are
  malformed/empty cue blocks. They were not repaired or counted as successes.
- All generated SRTs were parsed again; cue counts, text correspondence, positive
  durations and exactly the reported timing edits were checked. Source hashes
  were checked against the frozen run inputs.
- The actual `--observe-only` CLI was run on the old Collection Alass file:
  proposed `[267]`, corrected `[]`, all 426 parsed cues unchanged.
- Ruff passes on the changed Python files. No commit or push was performed.

## Artifacts and reproduction

`validation-20260929/results.json` and per-case folders contain the penalty-7
results, anchors, SRTs, logs and review candidates. The run used Alass 2.0.0's
default penalty 7; the runner now passes that value explicitly for reproducibility.
`validation-20260929/split-penalty-0.5/` contains the second setting comparison.
`legacy-alass-review/` inside the run preserves copies of the original Alass
files and their reviewed decisions. Original cache files were read-only.

Use a fresh run directory:

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python tools/semantic_sync_poc/expanded_validation.py \
  --run-dir tools/semantic_sync_poc/validation-next \
  --alass-bin /home/jimmyzourx/.cargo/bin/alass-cli --split-penalty 7

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/semantic_sync_poc/compare_alass_settings.py \
  --run-dir tools/semantic_sync_poc/validation-next \
  --alass-bin /home/jimmyzourx/.cargo/bin/alass-cli --split-penalty 0.5
```

For experimental monitoring, add `--observe-only` to `semantic_guard.py`.
This produces proposals and an unchanged-timing Alass copy. It does not install
a background monitor or change the production AutoSync path.
