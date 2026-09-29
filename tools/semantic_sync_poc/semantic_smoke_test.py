from pathlib import Path
import re

import numpy as np
from sentence_transformers import SentenceTransformer

from app.services.sync_service import _normalize


MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

ARABIC_QUERY = "أعلم أنك تفتقدين أمك"

ENGLISH_FILE = Path(
    "subs_cache/references/"
    "tt1748227_8ae53ebaa27a07affa2e_subdl_edition.srt"
)


def extract_text(path: Path):
    normalized, _ = _normalize(path.read_bytes())
    text = normalized.decode("utf-8-sig")

    cues = []

    for block in re.split(r"\n\s*\n", text.strip()):
        lines = block.splitlines()

        if len(lines) < 3:
            continue

        body = " ".join(
            line.strip()
            for line in lines[2:]
            if line.strip()
        )

        cues.append(body)

    return cues


print("Loading model...")
model = SentenceTransformer(MODEL, device="cpu")

english = extract_text(ENGLISH_FILE)[:10]

sentences = [ARABIC_QUERY] + english

embeddings = model.encode(
    sentences,
    normalize_embeddings=True,
    show_progress_bar=False,
)

query = embeddings[0]
candidates = embeddings[1:]

scores = candidates @ query

ranking = np.argsort(scores)[::-1]

print()
print("ARABIC QUERY:")
print(ARABIC_QUERY)

print()
print("TOP MATCHES:")

for rank, idx in enumerate(ranking[:5], 1):
    print(
        f"{rank}. score={scores[idx]:.4f} "
        f"EN #{idx + 1}: {english[idx]}"
    )
