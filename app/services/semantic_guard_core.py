"""Semantic Guard core algorithms - shared production implementation.

Reuses proven algorithms from tools/semantic_sync_poc without sys.path hacks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_STAMP = r"(\d{1,6}):([0-5]\d):([0-5]\d)[.,](\d{1,3})"
_SPAN = re.compile(rf"^{_STAMP}\s*-{{1,2}}>-?\s*{_STAMP}(.*)$")
_TAGS = re.compile(r"<[^>]+>|\{\\[^}]+\}")


@dataclass(frozen=True)
class Cue:
    number: str
    start: float
    end: float
    body: str
    suffix: str = ""

    @property
    def text(self) -> str:
        return " ".join(_TAGS.sub(" ", self.body).split())


def parse_srt(path_or_bytes: Path | bytes) -> list[Cue]:
    """Parse SRT bytes or path into Cue objects."""
    if isinstance(path_or_bytes, bytes):
        text = path_or_bytes.decode("utf-8-sig")
    else:
        text = path_or_bytes.read_text(encoding="utf-8-sig")

    if "\x00" in text or not text.strip():
        raise ValueError("Empty or invalid SRT")

    cues = []
    for index, block in enumerate(re.split(r"\n[ \t]*\n", text.strip()), 1):
        lines = block.strip().splitlines()
        match = _SPAN.fullmatch(lines[1].strip()) if len(lines) >= 3 else None
        if not match or not lines[0].strip():
            raise ValueError(f"Malformed SRT block {index}")
        groups = match.groups()

        def seconds(values):
            h, m, s = map(int, values[:3])
            return h * 3600 + m * 60 + s + int(values[3]) / 10 ** len(values[3])

        start, end = seconds(groups[:4]), seconds(groups[4:8])
        if end <= start:
            raise ValueError(f"Non-positive duration at block {index}")
        cues.append(Cue(str(index), start, end, "\n".join(lines[2:]), groups[8]))
    return cues


def anchors_from_scores(scores: np.ndarray, min_score: float = 0.75, min_margin: float = 0.15) -> list[tuple[int, int, float, float]]:
    """Find trusted monotonic anchors from embedding similarity scores."""
    if min(scores.shape) < 2 or not np.isfinite(scores).all():
        return []
    best_en, best_ar = scores.argmax(axis=1), scores.argmax(axis=0)
    row_top = np.partition(scores, -2, axis=1)[:, -2:]
    col_top = np.partition(scores, -2, axis=0)[-2:, :]
    candidates = []
    for ai, ei in enumerate(best_en):
        score = float(scores[ai, ei])
        margin = float(min(row_top[ai, 1] - row_top[ai, 0], col_top[1, ei] - col_top[0, ei]))
        if best_ar[ei] == ai and score >= min_score and margin >= min_margin:
            candidates.append((ai, int(ei), score, margin))
    weights, previous = [], []
    for i, (_, ei, score, margin) in enumerate(candidates):
        weights.append(score + margin)
        previous.append(-1)
        for j in range(i):
            value = weights[j] + score + margin
            if candidates[j][1] < ei and value > weights[i]:
                weights[i], previous[i] = value, j
    chain = []
    if weights:
        position = int(np.argmax(weights))
        while position != -1:
            chain.append(candidates[position])
            position = previous[position]
    return chain[::-1]


def robust_fit(x: np.ndarray, y: np.ndarray):
    """Deterministic median pairwise slopes, then MAD-trimmed least squares."""
    if len(x) < 3 or np.ptp(x) < 1:
        raise ValueError("Insufficient distinct anchor times")
    left, right = np.triu_indices(len(x), 1)
    usable = x[right] - x[left] > 1
    slope = np.median((y[right[usable]] - y[left[usable]]) / (x[right[usable]] - x[left[usable]]))
    offset = np.median(y - slope * x)
    for _ in range(5):
        residual = y - (slope * x + offset)
        median = np.median(residual)
        scale = 1.4826 * np.median(np.abs(residual - median))
        mask = np.abs(residual - median) <= max(0.5, 4 * scale)
        if mask.sum() < 3 or np.ptp(x[mask]) < 1:
            raise ValueError("Insufficient timing inliers")
        slope, offset = np.polyfit(x[mask], y[mask], 1)
    if not np.isfinite([slope, offset]).all() or slope <= 0:
        raise ValueError("Invalid timing model")
    return float(slope), float(offset), mask


def stats(errors: np.ndarray):
    values = np.abs(errors)
    return dict(
        mae=float(np.mean(values)),
        median=float(np.median(values)),
        p90=float(np.percentile(values, 90)),
        maximum=float(np.max(values))
    )


def guard_analyze(arabic: list[Cue], reference: list[Cue], alass: list[Cue], anchors: list[tuple[int, int, float, float]], *, observe_only: bool = True) -> dict[str, Any]:
    """Analyze semantic guard observation without modifying timings."""
    if len(arabic) != len(alass):
        raise ValueError("Arabic/Alass cue count mismatch")
    for i, (original, aligned) in enumerate(zip(arabic, alass, strict=True), 1):
        if " ".join(original.body.split()) != " ".join(aligned.body.split()):
            raise ValueError(f"Arabic/Alass text or order mismatch at block {i}")

    report: dict[str, Any] = {
        "trusted_anchors": len(anchors),
        "confident": False,
        "corrected": [],
        "mode": "observe-only" if observe_only else "guarded",
        "proposed_corrections": [],
        "suspicious": [],
        "reason": "Fewer than 20 trusted anchors"
    }

    if len(anchors) < 20:
        return report

    x = np.array([arabic[a[0]].start for a in anchors])
    y = np.array([reference[a[1]].start for a in anchors])

    try:
        slope, offset, mask = robust_fit(x, y)
        held_out = np.empty(len(x))
        for fold in range(5):
            test = np.arange(len(x)) % 5 == fold
            a, b, _ = robust_fit(x[~test], y[~test])
            held_out[test] = a * x[test] + b - y[test]
    except ValueError as exc:
        report["reason"] = str(exc)
        return report

    extent = max(c.end for c in arabic) - min(c.start for c in arabic)
    coverage = float(np.ptp(x[mask]) / max(extent, 1))
    cv = stats(held_out)
    confident = bool(
        mask.sum() >= 20
        and mask.mean() >= 0.8
        and coverage >= 0.6
        and cv["median"] <= 0.5
        and cv["p90"] <= 1.0
    )

    threshold = max(5.0, 4 * cv["p90"])
    report.update(
        slope=slope,
        offset_seconds=offset,
        timing_inliers=int(mask.sum()),
        coverage=coverage,
        confident=confident,
        threshold_seconds=threshold,
        reason="Confidence checks passed" if confident else "Confidence checks failed",
        semantic_cv_seconds=cv,
        original_seconds=stats(x - y),
        alass_seconds=stats(np.array([alass[a[0]].start for a in anchors]) - y)
    )

    support = {
        a[0]: (float(y[i]), bool(mask[i]), float(held_out[i]))
        for i, a in enumerate(anchors)
    }

    for i, (original, aligned) in enumerate(zip(arabic, alass, strict=True)):
        start, end = slope * original.start + offset, slope * original.end + offset
        gap = max(abs(aligned.start - max(0, start)), abs(aligned.end - max(0, end)))
        if gap < threshold:
            continue
        target, inlier, cv_error = support.get(i, (None, False, float("inf")))
        reason = "No reliable direct semantic support"
        repair = (
            confident
            and inlier
            and abs(cv_error) <= 1
            and x[mask].min() <= original.start <= x[mask].max()
            and end > max(0, start)
            and target is not None
            and abs(aligned.start - target) >= threshold
            and abs(start - target) <= 1
        )
        if repair:
            report["proposed_corrections"].append(i + 1)
            reason = "Alass start outlier confirmed by held-out semantic anchor"
        elif not confident:
            reason = "Model confidence insufficient"

        report["suspicious"].append({
            "block": i + 1,
            "disagreement_seconds": gap,
            "corrected": False,
            "proposed": bool(repair),
            "reason": reason
        })

    return report
