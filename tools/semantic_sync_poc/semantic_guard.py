#!/usr/bin/env python3
"""Experimental semantic timing guard. Never imports or changes production code."""

import argparse
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
STAMP = r"(\d{1,6}):([0-5]\d):([0-5]\d)[.,](\d{1,3})"
SPAN = re.compile(rf"^{STAMP}\s*-{{1,2}}>-?\s*{STAMP}(.*)$")
TAGS = re.compile(r"<[^>]+>|\{\\[^}]+\}")


@dataclass(frozen=True)
class Cue:
    number: str
    start: float
    end: float
    body: str
    suffix: str = ""

    @property
    def text(self):
        return " ".join(TAGS.sub(" ", self.body).split())


def parse(path):
    text = path.read_text(encoding="utf-8-sig")
    if "\x00" in text or not text.strip():
        raise ValueError(f"Empty or invalid SRT: {path}")
    cues = []
    for index, block in enumerate(re.split(r"\n[ \t]*\n", text.strip()), 1):
        lines = block.strip().splitlines()
        match = SPAN.fullmatch(lines[1].strip()) if len(lines) >= 3 else None
        if not match or not lines[0].strip():
            raise ValueError(f"Malformed SRT block {index}: {path}")
        groups = match.groups()

        def seconds(values):
            h, m, s = map(int, values[:3])
            return h * 3600 + m * 60 + s + int(values[3]) / 10 ** len(values[3])

        start, end = seconds(groups[:4]), seconds(groups[4:8])
        if end <= start:
            raise ValueError(f"Non-positive duration at block {index}: {path}")
        # Reference releases sometimes use labels such as "10th".
        cues.append(Cue(str(index), start, end, "\n".join(lines[2:]), groups[8]))
    return cues


def fmt(value):
    milliseconds = max(0, round(value * 1000))
    hours, milliseconds = divmod(milliseconds, 3600000)
    minutes, milliseconds = divmod(milliseconds, 60000)
    seconds, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def render(cues):
    return "\n\n".join(
        f"{c.number}\n{fmt(c.start)} --> {fmt(c.end)}{c.suffix}\n{c.body}" for c in cues
    ) + "\n"


def validate_pair(arabic, alass):
    if len(arabic) != len(alass):
        raise ValueError("Arabic/Alass cue count mismatch")
    for i, (original, aligned) in enumerate(zip(arabic, alass, strict=True), 1):
        if " ".join(original.body.split()) != " ".join(aligned.body.split()):
            raise ValueError(f"Arabic/Alass text or order mismatch at block {i}")


def anchors_from_scores(scores, min_score=0.75, min_margin=0.15):
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
    # Maximum-weight strictly monotonic chain, as in trusted_anchors.py.
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


def robust_fit(x, y):
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


def stats(errors):
    values = np.abs(errors)
    return dict(mae=float(np.mean(values)), median=float(np.median(values)),
                p90=float(np.percentile(values, 90)), maximum=float(np.max(values)))


def guard(arabic, reference, alass, anchors, *, observe_only=False):
    validate_pair(arabic, alass)
    report = {"trusted_anchors": len(anchors), "confident": False, "corrected": [],
              "mode": "observe-only" if observe_only else "guarded",
              "proposed_corrections": [],
              "suspicious": [], "reason": "Fewer than 20 trusted anchors"}
    if len(anchors) < 20:
        return alass, report
    x = np.array([arabic[a[0]].start for a in anchors])
    y = np.array([reference[a[1]].start for a in anchors])
    try:
        slope, offset, mask = robust_fit(x, y)
        held_out = np.empty(len(x))
        # Timing outliers are filtered within each training fold only.
        for fold in range(5):
            test = np.arange(len(x)) % 5 == fold
            a, b, _ = robust_fit(x[~test], y[~test])
            held_out[test] = a * x[test] + b - y[test]
    except ValueError as exc:
        report["reason"] = str(exc)
        return alass, report
    extent = max(c.end for c in arabic) - min(c.start for c in arabic)
    coverage = float(np.ptp(x[mask]) / max(extent, 1))
    cv = stats(held_out)
    confident = bool(mask.sum() >= 20 and mask.mean() >= 0.8 and coverage >= 0.6
                     and cv["median"] <= 0.5 and cv["p90"] <= 1.0)
    threshold = max(5.0, 4 * cv["p90"])
    report.update(slope=slope, offset_seconds=offset, timing_inliers=int(mask.sum()),
                  coverage=coverage, confident=confident, threshold_seconds=threshold,
                  reason="Confidence checks passed" if confident else "Confidence checks failed",
                  semantic_cv_seconds=cv, original_seconds=stats(x - y),
                  alass_seconds=stats(np.array([alass[a[0]].start for a in anchors]) - y))
    output = list(alass)
    # Scan every cue; disagreement alone is not proof Alass is wrong.
    # Repair only directly supported anchors, inside the fitted time range.
    support = {a[0]: (float(y[i]), bool(mask[i]), float(held_out[i]))
               for i, a in enumerate(anchors)}
    for i, (original, aligned) in enumerate(zip(arabic, alass, strict=True)):
        start, end = slope * original.start + offset, slope * original.end + offset
        gap = max(abs(aligned.start - max(0, start)), abs(aligned.end - max(0, end)))
        if gap < threshold:
            continue
        target, inlier, cv_error = support.get(i, (None, False, float("inf")))
        reason = "No reliable direct semantic support"
        repair = (confident and inlier and abs(cv_error) <= 1
                  and x[mask].min() <= original.start <= x[mask].max()
                  and end > max(0, start) and target is not None
                  and abs(aligned.start - target) >= threshold
                  and abs(start - target) <= 1)
        if repair:
            report["proposed_corrections"].append(i + 1)
            if not observe_only:
                output[i] = replace(aligned, start=max(0, start), end=end)
                report["corrected"].append(i + 1)
            reason = "Alass start outlier confirmed by held-out semantic anchor"
        elif not confident:
            reason = "Model confidence insufficient"
        report["suspicious"].append(dict(block=i + 1, disagreement_seconds=gap,
                                         corrected=bool(repair and not observe_only),
                                         proposed=bool(repair), reason=reason))
    return output, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("arabic", "reference", "alass", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--observe-only", action="store_true",
                        help="Report proposed corrections but preserve all Alass timings")
    parser.add_argument("--report", type=Path, help="JSON report; default OUTPUT.json")
    args = parser.parse_args()
    report_path = args.report or Path(str(args.output) + ".json")
    inputs = [args.arabic.resolve(), args.reference.resolve(), args.alass.resolve()]
    destinations = [args.output.resolve(), report_path.resolve()]
    if len(set(destinations)) != 2 or any(p in inputs for p in destinations):
        parser.error("Output/report must be distinct from each other and all inputs")
    if any(p.exists() for p in destinations):
        parser.error("Refusing to overwrite an existing output/report")
    try:
        arabic, reference, alass = map(parse, (args.arabic, args.reference, args.alass))
        validate_pair(arabic, alass)
        from sentence_transformers import SentenceTransformer

        print("Loading semantic model...", flush=True)
        encoder = SentenceTransformer(args.model, device="cpu")
        vectors = encoder.encode([c.text for c in arabic + reference],
                                 normalize_embeddings=True, show_progress_bar=False)
        scores = vectors[:len(arabic)] @ vectors[len(arabic):].T
        # Empty/format-only cues never become semantic anchors; indices stay intact.
        scores[[not c.text for c in arabic], :] = -1
        scores[:, [not c.text for c in reference]] = -1
        anchors = anchors_from_scores(scores)
        output, report = guard(arabic, reference, alass, anchors, observe_only=args.observe_only)
        report.update(model=args.model, cue_count=len(output),
                      anchors=[dict(arabic_block=a + 1, reference_block=e + 1,
                                    score=s, margin=m) for a, e, s, m in anchors])
        # Low confidence is an explicit pass-through, never a whole-file fallback fit.
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(render(output))
        with report_path.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps({k: v for k, v in report.items() if k != "anchors"}, indent=2))
        print(f"Wrote {args.output} and {report_path}")
    except (ValueError, OSError) as exc:
        parser.exit(2, f"semantic_guard: {exc}\n")


if __name__ == "__main__":
    main()
