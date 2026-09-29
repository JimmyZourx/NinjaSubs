#!/usr/bin/env python3
"""Reproducible local-cache experiment; writes only below --run-dir."""

import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
from semantic_guard import MODEL, anchors_from_scores, guard, parse, render, stats, validate_pair

# Fixture selection is metadata-based, fixed before seeing semantic results.
CASES = [
    ("sunshine-bluray", "tt0448134", "6519e11c00ee4b7f", "tt0448134_97ed92289670e947adaa_subdl_edition"),
    ("sunshine-dvd", "tt0448134", "de85beb3fa116284", "tt0448134_97ed92289670e947adaa_subdl_edition"),
    ("deuce-bigalow", "tt0205000", "166dfd052f16ed9b", "tt0205000_722cff1fa9bf3391cf71_subdl_edition"),
    ("ip-man-3", "tt2888046", "78e013d040a33cdf", "tt2888046_e666cee5bebc624f3532_subdl_edition"),
    ("tt0120591", "tt0120591", "c754623ba0a4d549", "tt0120591_4af463d6935e2f54a1c1_subdl_edition"),
    ("the-collection", "tt1748227", "43d229491b6048d2", "tt1748227_8ae53ebaa27a07affa2e_subdl_edition"),
    ("better-tomorrow", "tt0092263", "de8099e661d23434", "tt0092263_52346554f475b29d48e9_subdl_team"),
]


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(before, after, report):
    validate_pair(before, after)
    changed = [i + 1 for i, (a, b) in enumerate(zip(before, after, strict=True))
               if abs(a.start - b.start) > 0.00051 or abs(a.end - b.end) > 0.00051]
    assert changed == report["corrected"], (changed, report["corrected"])
    assert all(c.start >= 0 and c.end > c.start for c in after)
    return {"cue_count": len(after), "changed_blocks": changed, "valid": True}


def review_rows(arabic, reference, alass, scores, anchors, report):
    rows = []
    anchor_map = {a[0]: a[1] for a in anchors}
    for item in report["suspicious"]:
        i = item["block"] - 1
        predicted = [report["slope"] * t + report["offset_seconds"]
                     for t in (arabic[i].start, arabic[i].end)]
        top = np.argsort(scores[i])[-3:][::-1]
        rows.append({**item, "arabic": arabic[i].body,
                     "original": [arabic[i].start, arabic[i].end],
                     "alass": [alass[i].start, alass[i].end], "semantic": predicted,
                     "trusted_anchor": i in anchor_map,
                     "top_reference_candidates": [
                         {"block": int(j) + 1, "text": reference[j].body,
                          "start": reference[j].start, "end": reference[j].end,
                          "score": float(scores[i, j]),
                          "alass_start_error": alass[i].start - reference[j].start,
                          "model_start_error": predicted[0] - reference[j].start}
                         for j in top]})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--alass-bin", type=Path, required=True)
    parser.add_argument("--split-penalty", type=float, default=7.0)
    args = parser.parse_args()
    experiment = Path(__file__).resolve().parent
    run = args.run_dir.resolve()
    if not run.is_relative_to(experiment) or run == experiment:
        parser.error("Run directory must be a new child of tools/semantic_sync_poc")
    if not 0 < args.split_penalty <= 1000:
        parser.error("Require 0 < split penalty <= 1000")
    run.mkdir(exist_ok=False)
    save_json(run / "manifest.json", {"model": MODEL, "cases": CASES,
                                     "split_penalty": args.split_penalty,
                                     "selection": "Fixed cached pairs; no post-result cherry-picking",
                                     "production_changes": False})
    # All inputs remain read-only. Canonical reference copies are local to the run.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(2)
    encoder = SentenceTransformer(MODEL, device="cpu")
    vectors = {}

    def encode(cues):
        texts = [c.text for c in cues]
        key = hashlib.sha256(json.dumps(texts).encode()).hexdigest()
        if key not in vectors:
            vectors[key] = encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return vectors[key]

    results, completed = [], {}
    for name, imdb, arabic_id, reference_id in CASES:
        case = run / name
        case.mkdir()
        ar_path = Path("subs_cache") / (arabic_id + ".srt")
        ref_path = Path("subs_cache/references") / (reference_id + ".srt")
        row = {"name": name, "imdb": imdb, "arabic": str(ar_path), "reference": str(ref_path),
               "input_sha256": {str(p): fingerprint(p) for p in (ar_path, ref_path)}}
        print(f"START {name}", flush=True)
        try:
            arabic, reference = parse(ar_path), parse(ref_path)
            normalized = case / "reference.srt"
            normalized.write_text(render(reference), encoding="utf-8")
            alass_path = case / "alass.srt"
            with (case / "alass.log").open("w") as log:
                subprocess.run([str(args.alass_bin), str(normalized), str(ar_path), str(alass_path),
                                "--split-penalty", str(args.split_penalty)],
                               stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
            alass = parse(alass_path)
            validate_pair(arabic, alass)
            scores = encode(arabic) @ encode(reference).T
            scores[[not c.text for c in arabic], :] = -1
            scores[:, [not c.text for c in reference]] = -1
            anchors = anchors_from_scores(scores)
            output, report = guard(arabic, reference, alass, anchors)
            output_path = case / "guarded.srt"
            output_path.write_text(render(output), encoding="utf-8")
            row.update(report)
            row["output_validation"] = verify(alass, parse(output_path), report)
            if anchors:
                row["guarded_anchor_seconds"] = stats(np.array([
                    output[a].start - reference[e].start for a, e, _, _ in anchors]))
            # Pre-aligned Alass as source: a stability check, not proof of correct timing.
            stable, stable_report = guard(alass, reference, alass, anchors)
            row["real_alass_stability"] = {"confident": stable_report["confident"],
                                           "corrections": stable_report["corrected"],
                                           "reason": stable_report["reason"]}
            # Exact reference timing at anchor positions: controlled positive control.
            # The text/embedding matches stay fixed; this is explicitly synthetic.
            exact = list(reference)
            synthetic_ar = list(alass)
            for a, e, _, _ in anchors:
                synthetic_ar[a] = replace(synthetic_ar[a], start=exact[e].start, end=exact[e].end)
            unchanged, positive = guard(synthetic_ar, reference, synthetic_ar, anchors)
            assert unchanged == synthetic_ar and not positive["corrected"]
            row["synthetic_correct_timing_control"] = {"unchanged": True,
                                                        "confident": positive["confident"]}
            save_json(case / "review.json", review_rows(arabic, reference, alass, scores, anchors, report))
            save_json(case / "anchors.json", anchors)
            completed[name] = (arabic, reference, alass)
            row["status"] = "completed"
        except (ValueError, OSError, subprocess.SubprocessError, AssertionError) as exc:
            row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        assert all(fingerprint(Path(p)) == digest for p, digest in row["input_sha256"].items())
        save_json(case / "result.json", row)
        results.append(row)
        save_json(run / "results.json", results)
        print(json.dumps({k: row[k] for k in ("name", "status", "trusted_anchors", "confident", "corrected", "error") if k in row}), flush=True)

    # Cross-film semantic negative control; never claim an unrelated reference works.
    if "sunshine-bluray" in completed and "deuce-bigalow" in completed:
        arabic, _, alass = completed["sunshine-bluray"]
        _, wrong_reference, _ = completed["deuce-bigalow"]
        wrong_scores = encode(arabic) @ encode(wrong_reference).T
        wrong_anchors = anchors_from_scores(wrong_scores)
        output, report = guard(arabic, wrong_reference, alass, wrong_anchors)
        report["unchanged"] = output == alass
        save_json(run / "wrong-film-control.json", report)
        assert not report["confident"] and output == alass

    invalid = []
    for name in ("0cd84199ddfca831", "c45a188c34e44af9", "faaae04f7b3ecc25"):
        path = Path("subs_cache") / (name + ".srt")
        try:
            parse(path)
            invalid.append({"path": str(path), "rejected": False})
        except ValueError as exc:
            invalid.append({"path": str(path), "rejected": True, "reason": str(exc)})
    save_json(run / "invalid-inputs.json", invalid)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
