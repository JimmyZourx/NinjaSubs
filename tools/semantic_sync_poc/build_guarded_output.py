from pathlib import Path
import re

SLOPE = 1.04299092
OFFSET_MS = -796.0

ORIGINAL = Path("subs_cache/43d229491b6048d2.srt")
ALASS = Path("/tmp/the-collection.alass-sync.srt")
OUTPUT = Path("/tmp/the-collection.guarded-sync.srt")

SUSPICIOUS = {267}

SPAN = re.compile(
    r"^(\d{2,6}):([0-5]\d):([0-5]\d),(\d{3})"
    r" --> "
    r"(\d{2,6}):([0-5]\d):([0-5]\d),(\d{3})$"
)


def to_ms(groups):
    h, m, s, ms = map(int, groups)
    return (
        h * 3_600_000
        + m * 60_000
        + s * 1_000
        + ms
    )


def fmt(ms):
    ms = max(0, round(ms))

    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, milli = divmod(rem, 1_000)

    return f"{h:02}:{m:02}:{s:02},{milli:03}"


def parse_blocks(path):
    text = path.read_text(encoding="utf-8-sig")
    return re.split(r"\n\s*\n", text.strip())


original_blocks = parse_blocks(ORIGINAL)
alass_blocks = parse_blocks(ALASS)

if len(original_blocks) != len(alass_blocks):
    raise SystemExit(
        f"Block mismatch: original={len(original_blocks)} "
        f"alass={len(alass_blocks)}"
    )

out = []

for cue_no, (orig, ala) in enumerate(
    zip(original_blocks, alass_blocks),
    1,
):
    if cue_no not in SUSPICIOUS:
        out.append(ala)
        continue

    orig_lines = orig.splitlines()
    ala_lines = ala.splitlines()

    if len(orig_lines) < 3 or len(ala_lines) < 3:
        raise ValueError(f"Invalid cue #{cue_no}")

    m = SPAN.match(orig_lines[1].strip())

    if not m:
        raise ValueError(
            f"Invalid timestamp cue #{cue_no}: {orig_lines[1]}"
        )

    g = m.groups()

    start = to_ms(g[:4])
    end = to_ms(g[4:])

    new_start = SLOPE * start + OFFSET_MS
    new_end = SLOPE * end + OFFSET_MS

    # Keep Alass text/body; replace only timing.
    ala_lines[1] = (
        f"{fmt(new_start)} --> {fmt(new_end)}"
    )

    out.append("\n".join(ala_lines))

OUTPUT.write_text(
    "\n\n".join(out) + "\n",
    encoding="utf-8",
)

print("WROTE:", OUTPUT)
print("Guarded cues:", sorted(SUSPICIOUS))
