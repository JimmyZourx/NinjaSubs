import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from app.services.sync_service import _normalize

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

AR_FILE = Path("subs_cache/43d229491b6048d2.srt")
EN_FILE = Path(
    "subs_cache/references/"
    "tt1748227_8ae53ebaa27a07affa2e_subdl_edition.srt"
)

MIN_SCORE = 0.75
MIN_MARGIN = 0.15

TAG = re.compile(r"<[^>]+>|\{\\[^}]+\}")
SPACE = re.compile(r"\s+")

TIME = re.compile(
    r"^(\d{1,6}):([0-5]\d):([0-5]\d),(\d{3})"
)


def timestamp_ms(span: str) -> int:
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


def extract(path: Path):
    normalized, _ = _normalize(path.read_bytes())
    text = normalized.decode("utf-8-sig")

    cues = []

    for block in re.split(r"\n\s*\n", text.strip()):
        lines = block.splitlines()

        if len(lines) < 3:
            continue

        span = lines[1].strip()

        body = " ".join(
            line.strip()
            for line in lines[2:]
            if line.strip()
        )

        body = TAG.sub(" ", body)
        body = SPACE.sub(" ", body).strip()

        if not body:
            continue

        cues.append(
            {
                "time": span,
                "start": timestamp_ms(span),
                "text": body,
            }
        )

    return cues


ar = extract(AR_FILE)
en = extract(EN_FILE)

print("Arabic cues :", len(ar))
print("English cues:", len(en))
print()
print("Loading model...")

model = SentenceTransformer(MODEL, device="cpu")

ar_vec = model.encode(
    [x["text"] for x in ar],
    normalize_embeddings=True,
    show_progress_bar=True,
)

en_vec = model.encode(
    [x["text"] for x in en],
    normalize_embeddings=True,
    show_progress_bar=True,
)

scores = ar_vec @ en_vec.T


# Best English cue for each Arabic cue
best_en = np.argmax(scores, axis=1)

# Best Arabic cue for each English cue
best_ar = np.argmax(scores, axis=0)


def top_margin(row):
    order = np.argsort(row)[::-1]

    best = float(row[order[0]])
    second = float(row[order[1]])

    return best, best - second


candidates = []

for ai in range(len(ar)):
    ei = int(best_en[ai])

    # Mutual nearest-neighbour requirement
    if int(best_ar[ei]) != ai:
        continue

    score, ar_margin = top_margin(scores[ai])

    # Margin from English side too
    en_column = scores[:, ei]
    _, en_margin = top_margin(en_column)

    if score < MIN_SCORE:
        continue

    if ar_margin < MIN_MARGIN:
        continue

    if en_margin < MIN_MARGIN:
        continue

    candidates.append(
        {
            "ai": ai,
            "ei": ei,
            "score": score,
            "margin": min(ar_margin, en_margin),
        }
    )


print()
print("Mutual high-confidence candidates:", len(candidates))


# Weighted monotonic sequence:
# Arabic indices already increase.
# We choose an English-index sequence that also only increases.
n = len(candidates)

dp = [0.0] * n
prev = [-1] * n

for i in range(n):
    current = candidates[i]

    weight = (
        current["score"]
        + current["margin"]
    )

    dp[i] = weight

    for j in range(i):
        earlier = candidates[j]

        if earlier["ei"] >= current["ei"]:
            continue

        candidate_score = dp[j] + weight

        if candidate_score > dp[i]:
            dp[i] = candidate_score
            prev[i] = j


anchors = []

if candidates:
    pos = max(range(n), key=lambda i: dp[i])

    while pos != -1:
        anchors.append(candidates[pos])
        pos = prev[pos]

    anchors.reverse()


print("Trusted monotonic anchors      :", len(anchors))

if candidates:
    print(
        "Anchor retention              :",
        f"{len(anchors) / len(candidates) * 100:.1f}%"
    )

print()
print("=== TRUSTED ANCHORS ===")

for number, anchor in enumerate(anchors[:40], 1):
    ai = anchor["ai"]
    ei = anchor["ei"]

    delta = en[ei]["start"] - ar[ai]["start"]

    print()
    print(
        f"{number:02}. "
        f"AR #{ai+1} -> EN #{ei+1}  "
        f"score={anchor['score']:.4f}  "
        f"margin={anchor['margin']:.4f}  "
        f"delta={delta:+d} ms"
    )

    print("AR:", ar[ai]["text"])
    print("EN:", en[ei]["text"])

print()
print("=== SUMMARY ===")

if anchors:
    deltas = np.array(
        [
            en[a["ei"]]["start"]
            - ar[a["ai"]]["start"]
            for a in anchors
        ]
    )

    print("Anchors       :", len(anchors))
    print("First AR cue  :", anchors[0]["ai"] + 1)
    print("Last AR cue   :", anchors[-1]["ai"] + 1)
    print("Median delta  :", f"{np.median(deltas):+.0f} ms")
    print("Min delta     :", f"{np.min(deltas):+d} ms")
    print("Max delta     :", f"{np.max(deltas):+d} ms")
