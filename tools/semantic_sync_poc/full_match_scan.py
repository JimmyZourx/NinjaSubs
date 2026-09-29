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

TAG = re.compile(r"<[^>]+>|\{\\[^}]+\}")
SPACE = re.compile(r"\s+")


def extract(path: Path):
    normalized, _ = _normalize(path.read_bytes())
    text = normalized.decode("utf-8-sig")

    cues = []

    for block in re.split(r"\n\s*\n", text.strip()):
        lines = block.splitlines()

        if len(lines) < 3:
            continue

        timestamp = lines[1].strip()

        body = " ".join(
            x.strip()
            for x in lines[2:]
            if x.strip()
        )

        body = TAG.sub(" ", body)
        body = SPACE.sub(" ", body).strip()

        if body:
            cues.append((timestamp, body))

    return cues


ar = extract(AR_FILE)
en = extract(EN_FILE)

print(f"Arabic cues : {len(ar)}")
print(f"English cues: {len(en)}")
print()
print("Loading model...")

model = SentenceTransformer(MODEL, device="cpu")

ar_vec = model.encode(
    [x[1] for x in ar],
    normalize_embeddings=True,
    show_progress_bar=True,
)

en_vec = model.encode(
    [x[1] for x in en],
    normalize_embeddings=True,
    show_progress_bar=True,
)

# 426 x 429 تقريباً فقط — صغير جداً
scores = ar_vec @ en_vec.T

matches = []

for ar_idx in range(len(ar)):
    order = np.argsort(scores[ar_idx])[::-1]

    best = int(order[0])
    second = int(order[1])

    best_score = float(scores[ar_idx, best])
    second_score = float(scores[ar_idx, second])
    margin = best_score - second_score

    matches.append(
        (
            best_score,
            margin,
            ar_idx,
            best,
            second_score,
        )
    )

# نعرض أقوى النتائج مع أفضلية واضحة عن المركز الثاني
matches.sort(reverse=True)

print("\n=== TOP HIGH-CONFIDENCE MATCHES ===")

shown = 0

for score, margin, ai, ei, _second in matches:
    if score < 0.70:
        continue

    print()
    print(
        f"AR #{ai+1} -> EN #{ei+1}  "
        f"score={score:.4f}  margin={margin:.4f}"
    )
    print("AR TIME:", ar[ai][0])
    print("EN TIME:", en[ei][0])
    print("AR:", ar[ai][1])
    print("EN:", en[ei][1])

    shown += 1

    if shown == 25:
        break

print()
print("High-confidence shown:", shown)
