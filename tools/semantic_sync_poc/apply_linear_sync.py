from pathlib import Path
import re

SLOPE = 1.04299092
OFFSET_MS = -796.0

SOURCE = Path("subs_cache/43d229491b6048d2.srt")
OUTPUT = Path("/tmp/the-collection.semantic-sync.srt")

SPAN = re.compile(
    r"^(\d{2,6}):([0-5]\d):([0-5]\d),(\d{3})"
    r" --> "
    r"(\d{2,6}):([0-5]\d):([0-5]\d),(\d{3})$"
)


def to_ms(h, m, s, ms):
    return (
        int(h) * 3_600_000
        + int(m) * 60_000
        + int(s) * 1_000
        + int(ms)
    )


def fmt(ms):
    ms = max(0, round(ms))

    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, milli = divmod(rem, 1_000)

    return f"{h:02}:{m:02}:{s:02},{milli:03}"


text = SOURCE.read_text(encoding="utf-8-sig")

out = []

for line in text.splitlines():
    m = SPAN.match(line.strip())

    if not m:
        out.append(line)
        continue

    (
        h1, m1, s1, ms1,
        h2, m2, s2, ms2,
    ) = m.groups()

    start = to_ms(h1, m1, s1, ms1)
    end = to_ms(h2, m2, s2, ms2)

    new_start = SLOPE * start + OFFSET_MS
    new_end = SLOPE * end + OFFSET_MS

    if new_end <= new_start:
        raise ValueError("Invalid retimed cue")

    out.append(
        f"{fmt(new_start)} --> {fmt(new_end)}"
    )

OUTPUT.write_text(
    "\n".join(out) + "\n",
    encoding="utf-8",
)

print("SOURCE :", SOURCE)
print("OUTPUT :", OUTPUT)
print("SLOPE  :", SLOPE)
print("OFFSET :", f"{OFFSET_MS:+.0f} ms")
