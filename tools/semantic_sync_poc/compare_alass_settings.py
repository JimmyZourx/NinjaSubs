"""Re-evaluate frozen semantic anchors against another Alass split penalty."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from expanded_validation import fingerprint, save_json, verify
from semantic_guard import guard, parse, render, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--alass-bin", type=Path, required=True)
    parser.add_argument("--split-penalty", type=float, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    run = args.run_dir.resolve()
    if not run.is_relative_to(root) or run == root or not 0 < args.split_penalty <= 1000:
        parser.error("Require an experimental run directory and 0 < split penalty <= 1000")
    destination = run / f"split-penalty-{args.split_penalty:g}"
    destination.mkdir(exist_ok=False)
    results = []
    for original in json.loads((run / "results.json").read_text()):
        if original["status"] != "completed":
            continue
        name = original["name"]
        case = destination / name
        case.mkdir()
        row = {"name": name, "split_penalty": args.split_penalty}
        print(f"START {name}", flush=True)
        try:
            assert all(fingerprint(Path(p)) == digest
                       for p, digest in original["input_sha256"].items())
            arabic, reference = parse(Path(original["arabic"])), parse(Path(original["reference"]))
            anchors = json.loads((run / name / "anchors.json").read_text())
            alass_path = case / "alass.srt"
            command = [str(args.alass_bin), str(run / name / "reference.srt"),
                       original["arabic"], str(alass_path), "--split-penalty", str(args.split_penalty)]
            row["command"] = command
            with (case / "alass.log").open("w") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
            alass = parse(alass_path)
            output, report = guard(arabic, reference, alass, anchors)
            row.update(report)
            path = case / "guarded.srt"
            path.write_text(render(output), encoding="utf-8")
            row["output_validation"] = verify(alass, parse(path), report)
            row["guarded_anchor_seconds"] = stats(np.array([
                output[a].start - reference[e].start for a, e, _, _ in anchors]))
            observed, observation = guard(arabic, reference, alass, anchors, observe_only=True)
            assert observed == alass and not observation["corrected"]
            assert observation["proposed_corrections"] == report["corrected"]
            row["observe_only_verified"] = True
            row["status"] = "completed"
        except (ValueError, OSError, AssertionError, subprocess.SubprocessError) as exc:
            row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save_json(case / "result.json", row)
        results.append(row)
        save_json(destination / "results.json", results)
        print(json.dumps({k: row[k] for k in ("name", "status", "confident", "corrected", "error") if k in row}), flush=True)


if __name__ == "__main__":
    main()
