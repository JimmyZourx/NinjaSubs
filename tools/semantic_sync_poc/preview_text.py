import argparse
import re
from pathlib import Path

from app.services.sync_service import _normalize

ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
HTML_TAGS = re.compile(r"<[^>]+>")
ASS_TAGS = re.compile(r"\{\\[^}]+\}")
BIDI_MARKS = re.compile(r"[\u200e\u200f\u202a-\u202e\u2066-\u2069]")
WHITESPACE = re.compile(r"\s+")


def clean_text(text: str, arabic: bool) -> str:
    text = HTML_TAGS.sub(" ", text)
    text = ASS_TAGS.sub(" ", text)
    text = BIDI_MARKS.sub("", text)
    text = text.replace("\u0640", "")  # tatweel

    if arabic:
        text = ARABIC_DIACRITICS.sub("", text)

    return WHITESPACE.sub(" ", text).strip()


def extract_cues(path: Path, arabic: bool):
    normalized, _ = _normalize(path.read_bytes())
    text = normalized.decode("utf-8-sig")

    result = []

    for block in re.split(r"\n\s*\n", text.strip()):
        lines = block.splitlines()

        if len(lines) < 3:
            continue

        timestamp = lines[1].strip()
        original = " ".join(line.strip() for line in lines[2:] if line.strip())
        cleaned = clean_text(original, arabic)

        result.append((timestamp, original, cleaned))

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arabic", type=Path, required=True)
    parser.add_argument("--english", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    ar = extract_cues(args.arabic, True)
    en = extract_cues(args.english, False)

    print("=== ARABIC ===")
    for i, (ts, original, cleaned) in enumerate(ar[:args.limit], 1):
        print(f"\nAR #{i}  {ts}")
        print("RAW  :", original)
        print("CLEAN:", cleaned)

    print("\n\n=== ENGLISH ===")
    for i, (ts, original, cleaned) in enumerate(en[:args.limit], 1):
        print(f"\nEN #{i}  {ts}")
        print("RAW  :", original)
        print("CLEAN:", cleaned)


if __name__ == "__main__":
    main()
