import json
import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from app.services.sync_service import _normalize

IMDB = "tt0092263"
MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

MIN_SCORE = 0.75
MIN_MARGIN = 0.15

TAG = re.compile(r"<[^>]+>|\{\\[^}]+\}")
SPACE = re.compile(r"\s+")
TIME = re.compile(r"^(\d{1,6}):([0-5]\d):([0-5]\d),(\d{3})")


def timestamp_ms(span):
    m = TIME.match(span)
    if not m:
        raise ValueError(span)

    h, mi, s, ms = map(int, m.groups())

    return (
        h * 3_600_000
        + mi * 60_000
        + s * 1_000
        + ms
    )


def extract(path):
    normalized, _ = _normalize(path.read_bytes())
    text = normalized.decode("utf-8-sig")

    cues = []

    for block in re.split(r"\n\s*\n", text.strip()):
        lines = block.splitlines()

        if len(lines) < 3:
            continue

        body = " ".join(
            x.strip()
            for x in lines[2:]
            if x.strip()
        )

        body = TAG.sub(" ", body)
        body = SPACE.sub(" ", body).strip()

        if not body:
            continue

        cues.append({
            "start": timestamp_ms(lines[1].strip()),
            "text": body,
        })

    return cues


def margin(values):
    order = np.argsort(values)[::-1]

    if len(order) < 2:
        return 0.0

    return (
        float(values[order[0]])
        - float(values[order[1]])
    )


def trusted_anchors(scores):
    best_en = np.argmax(scores, axis=1)
    best_ar = np.argmax(scores, axis=0)

    candidates = []

    for ai in range(scores.shape[0]):
        ei = int(best_en[ai])

        if int(best_ar[ei]) != ai:
            continue

        score = float(scores[ai, ei])

        if score < MIN_SCORE:
            continue

        if margin(scores[ai]) < MIN_MARGIN:
            continue

        if margin(scores[:, ei]) < MIN_MARGIN:
            continue

        candidates.append((ai, ei, score))

    n = len(candidates)

    if not n:
        return []

    dp = [0.0] * n
    prev = [-1] * n

    for i, (_, ei, score) in enumerate(candidates):
        dp[i] = score

        for j in range(i):
            _, old_ei, _ = candidates[j]

            if old_ei >= ei:
                continue

            value = dp[j] + score

            if value > dp[i]:
                dp[i] = value
                prev[i] = j

    pos = max(range(n), key=lambda i: dp[i])

    anchors = []

    while pos != -1:
        anchors.append(candidates[pos])
        pos = prev[pos]

    anchors.reverse()

    return anchors


arabic_files = []

for meta in Path("subs_cache/_meta").glob("*.json"):
    try:
        d = json.loads(meta.read_text())
    except Exception:
        continue

    if d.get("imdb_id") != IMDB:
        continue

    if d.get("lang") != "ara":
        continue

    srt = Path("subs_cache") / f"{meta.stem}.srt"

    if not srt.exists():
        continue

    arabic_files.append({
        "id": meta.stem,
        "path": srt,
        "release": d.get("release_name"),
    })


english_files = sorted(
    Path("subs_cache/references").glob(f"{IMDB}_*.srt")
)

print("Arabic candidates :", len(arabic_files))
print("English references:", len(english_files))

if not arabic_files or not english_files:
    raise SystemExit("Missing Arabic candidates or English references")


print("\nLoading model...")

model = SentenceTransformer(MODEL, device="cpu")


all_files = {}

valid_arabic_files = []

for item in arabic_files:
    path = item["path"]

    try:
        cues = extract(path)
    except Exception as exc:
        print(
            "SKIP AR:",
            item["id"],
            type(exc).__name__,
            str(exc),
        )
        continue

    print("Encoding AR:", item["id"], len(cues), "cues")

    vectors = model.encode(
        [x["text"] for x in cues],
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    all_files[str(path)] = (cues, vectors)
    valid_arabic_files.append(item)

arabic_files = valid_arabic_files


for path in english_files:
    cues = extract(path)

    print("Encoding EN:", path.name, len(cues), "cues")

    vectors = model.encode(
        [x["text"] for x in cues],
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    all_files[str(path)] = (cues, vectors)


results = []

for ar_item in arabic_files:
    ar_cues, ar_vec = all_files[str(ar_item["path"])]

    for en_path in english_files:
        en_cues, en_vec = all_files[str(en_path)]

        scores = ar_vec @ en_vec.T
        anchors = trusted_anchors(scores)

        if len(anchors) < 10:
            results.append({
                "ar": ar_item,
                "en": en_path,
                "anchors": len(anchors),
                "r2": None,
                "median": None,
                "max": None,
                "slope": None,
                "offset": None,
            })
            continue

        x = np.array([
            ar_cues[ai]["start"] / 1000.0
            for ai, _, _ in anchors
        ])

        y = np.array([
            en_cues[ei]["start"] / 1000.0
            for _, ei, _ in anchors
        ])

        slope, offset = np.polyfit(x, y, 1)

        predicted = slope * x + offset
        residuals = y - predicted

        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        r2 = (
            1.0 - ss_res / ss_tot
            if ss_tot > 0
            else 0.0
        )

        results.append({
            "ar": ar_item,
            "en": en_path,
            "anchors": len(anchors),
            "r2": float(r2),
            "median": float(np.median(np.abs(residuals))),
            "max": float(np.max(np.abs(residuals))),
            "slope": float(slope),
            "offset": float(offset),
        })


def sort_key(r):
    if r["r2"] is None:
        return (-1, -1, -999)

    return (
        r["anchors"],
        r["r2"],
        -r["median"],
    )


results.sort(key=sort_key, reverse=True)


print("\n=== TOP PAIRS ===")

for rank, r in enumerate(results[:12], 1):
    print()
    print(f"#{rank}")

    print("AR ID      :", r["ar"]["id"])
    print("AR RELEASE :", r["ar"]["release"])
    print("EN REF     :", r["en"].name)
    print("ANCHORS    :", r["anchors"])

    if r["r2"] is None:
        print("FIT        : insufficient anchors")
        continue

    print("R²         :", f'{r["r2"]:.6f}')
    print("MEDIAN ERR :", f'{r["median"]:.3f} sec')
    print("MAX ERR    :", f'{r["max"]:.3f} sec')
    print("SLOPE      :", f'{r["slope"]:.8f}')
    print("OFFSET     :", f'{r["offset"]:+.3f} sec')
