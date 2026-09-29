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
ALASS_FILE = Path("/tmp/the-collection.alass-sync.srt")

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

        span = lines[1].strip()

        body = " ".join(
            line.strip()
            for line in lines[2:]
            if line.strip()
        )

        body = TAG.sub(" ", body)
        body = SPACE.sub(" ", body).strip()

        cues.append(
            {
                "start": timestamp_ms(span),
                "text": body,
            }
        )

    return cues


ar = extract(AR_FILE)
en = extract(EN_FILE)
alass = extract(ALASS_FILE)

if len(ar) != len(alass):
    raise SystemExit(
        f"Cue count mismatch: original={len(ar)} alass={len(alass)}"
    )

print("Arabic :", len(ar))
print("English:", len(en))
print("Alass  :", len(alass))
print()
print("Finding semantic anchors...")

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

best_en = np.argmax(scores, axis=1)
best_ar = np.argmax(scores, axis=0)


def margin(values):
    order = np.argsort(values)[::-1]
    return float(values[order[0]]) - float(values[order[1]])


candidates = []

for ai in range(len(ar)):
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


# Longest weighted monotonic chain
n = len(candidates)
dp = [0.0] * n
prev = [-1] * n

for i, (_, ei, score) in enumerate(candidates):
    dp[i] = score

    for j in range(i):
        _, prev_ei, _ = candidates[j]

        if prev_ei >= ei:
            continue

        value = dp[j] + score

        if value > dp[i]:
            dp[i] = value
            prev[i] = j


anchors = []

if candidates:
    pos = max(range(n), key=lambda x: dp[x])

    while pos != -1:
        anchors.append(candidates[pos])
        pos = prev[pos]

    anchors.reverse()


print("Trusted anchors:", len(anchors))

if len(anchors) < 20:
    raise SystemExit("Not enough anchors for cross-validation")


def report(name, errors):
    errors = np.asarray(errors, dtype=float)

    print(f"\n{name}")
    print("-" * len(name))
    print(f"MAE       : {np.mean(np.abs(errors)):.3f} sec")
    print(f"Median AE : {np.median(np.abs(errors)):.3f} sec")
    print(f"P90 AE    : {np.percentile(np.abs(errors), 90):.3f} sec")
    print(f"Max AE    : {np.max(np.abs(errors)):.3f} sec")


original_errors = []
alass_errors = []
semantic_errors = []

# Deterministic 5-fold timing cross-validation
for fold in range(5):
    train = [
        a for i, a in enumerate(anchors)
        if i % 5 != fold
    ]

    test = [
        a for i, a in enumerate(anchors)
        if i % 5 == fold
    ]

    x_train = np.array([
        ar[ai]["start"] / 1000.0
        for ai, _, _ in train
    ])

    y_train = np.array([
        en[ei]["start"] / 1000.0
        for _, ei, _ in train
    ])

    slope, intercept = np.polyfit(
        x_train,
        y_train,
        1,
    )

    for ai, ei, _ in test:
        target = en[ei]["start"] / 1000.0

        original_time = ar[ai]["start"] / 1000.0
        alass_time = alass[ai]["start"] / 1000.0

        semantic_time = (
            slope * original_time
            + intercept
        )

        original_errors.append(
            original_time - target
        )

        alass_errors.append(
            alass_time - target
        )

        semantic_errors.append(
            semantic_time - target
        )


print("\n=== HELD-OUT TIMING RESULTS ===")

report("ORIGINAL", original_errors)
report("ALASS", alass_errors)
report("SEMANTIC 5-FOLD CV", semantic_errors)

print("\nLower is better.")
