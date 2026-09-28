"""Bounded, cancellation-safe subtitle-only Alass execution."""

from __future__ import annotations

import asyncio
import math
import os
import re
import stat
import tempfile
import threading
from pathlib import Path

from app.config import settings
from app.extractor import MAX_SUBTITLE_ENTRY_BYTES
from app.utils.ass_converter import convert_ass_to_srt

STREAM_LIMIT = 64 * 1024
TERMINATE_GRACE_SECONDS = 0.5

_ALASS_GATE = threading.BoundedSemaphore(max(1, settings.ALASS_MAX_CONCURRENCY))

_STAMP = r"(\d{2,6}):([0-5]\d):([0-5]\d),(\d{3})"
_SPAN = re.compile(rf"^{_STAMP} --> {_STAMP}$")


def _parse_srt(text: str) -> list[tuple[int, int, str]]:
    """Parse a well-formed SRT string into (start_ms, end_ms, body_text) tuples."""
    cues: list[tuple[int, int, str]] = []
    for block in re.split(r"\n[ \t]*\n", text.strip()):
        lines = block.splitlines()
        if len(lines) < 3 or not re.fullmatch(r"\d{1,9}", lines[0]):
            raise ValueError("invalid cue")
        match = _SPAN.fullmatch(lines[1])
        if match is None:
            raise ValueError("invalid timestamp")
        values = list(map(int, match.groups()))
        start = sum(a * b for a, b in zip(values[:4], (3600000, 60000, 1000, 1), strict=True))
        end = sum(a * b for a, b in zip(values[4:], (3600000, 60000, 1000, 1), strict=True))
        body = "\n".join(lines[2:])
        if start >= end or not body.strip() or "\x00" in body:
            raise ValueError("invalid cue")
        cues.append((start, end, body))
    return cues


def _vtt_to_srt(text: str) -> str:
    """Convert WebVTT to strict SubRip format."""
    blocks: list[str] = []
    for block in re.split(r"\n[ \t]*\n", text.strip()):
        lines = block.splitlines()
        if not lines:
            continue
        if lines[0].startswith(("WEBVTT", "NOTE", "STYLE", "REGION")):
            continue
        if "-->" not in lines[0]:
            lines = lines[1:]
        if len(lines) < 2:
            raise ValueError("invalid VTT")
        match = re.fullmatch(
            r"((?:\d{2,6}:)?[0-5]\d:[0-5]\d\.\d{3})\s+-->\s+"
            r"((?:\d{2,6}:)?[0-5]\d:[0-5]\d\.\d{3})(?:[ \t]+[^\n]*)?",
            lines[0],
        )
        if match is None:
            raise ValueError("invalid VTT timestamp")
        stamps = [
            ("00:" + t if t.count(":") == 1 else t).replace(".", ",")
            for t in match.groups()
        ]
        blocks.append(f"{len(blocks) + 1}\n{stamps[0]} --> {stamps[1]}\n" + "\n".join(lines[1:]))
    return "\n\n".join(blocks) + "\n"


def _normalize(data: bytes) -> tuple[bytes, list[tuple[int, int, str]]]:
    """Normalize subtitle bytes to SRT and return (normalized_bytes, parsed_cues).

    Raises ValueError on size, null bytes, or unparseable format.
    """
    if not data or len(data) > MAX_SUBTITLE_ENTRY_BYTES:
        raise ValueError("input size")
    text = data.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    if "\x00" in text:
        raise ValueError("invalid text")
    if "[events]" in text.lower():
        text = convert_ass_to_srt(text, apply_rtl=False)
    elif text.startswith("WEBVTT"):
        text = _vtt_to_srt(text)
    normalized = text.encode("utf-8")
    if len(normalized) > MAX_SUBTITLE_ENTRY_BYTES:
        raise ValueError("normalized size")
    return normalized, _parse_srt(text)


def _duration(cues: list[tuple[int, int, str]]) -> int:
    return max(end for _, end, _ in cues) - min(start for start, _, _ in cues)


def _compatible(a: list[tuple[int, int, str]], b: list[tuple[int, int, str]]) -> bool:
    """Two cue lists are duration-compatible if their runtimes differ by at most 2x."""
    if not a or not b:
        return False
    return 0.5 <= _duration(a) / _duration(b) <= 2.0


def _read_output(path: Path) -> bytes:
    """Read the output file safely: reject non-regular files, symlinks, FIFOs, and oversize."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= MAX_SUBTITLE_ENTRY_BYTES:
            raise ValueError("output size or type")
        output = stream.read(MAX_SUBTITLE_ENTRY_BYTES + 1)
    if len(output) > MAX_SUBTITLE_ENTRY_BYTES:
        raise ValueError("output size")
    return output


async def _drain(stream: asyncio.StreamReader, stop: asyncio.Event) -> None:
    """Read stream chunks until STREAM_LIMIT (64 KiB) is reached or stream ends.

    Retains no diagnostics; only one 4096-byte chunk count, including after overflow.
    """
    total = 0
    try:
        while chunk := await stream.read(4096):
            total = min(STREAM_LIMIT + 1, total + len(chunk))
            if total > STREAM_LIMIT:
                stop.set()
    except Exception:
        stop.set()
        raise


async def _reap(proc: asyncio.subprocess.Process) -> None:
    """Terminate then kill the process, waiting for it to fully exit."""
    if proc.returncode is None:
        try:
            proc.terminate()
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(proc.wait(), TERMINATE_GRACE_SECONDS)
        except TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
    await proc.wait()


class SubtitleSyncService:
    """Return normalized synced SRT bytes or None; never log subtitle content."""

    def __init__(self) -> None:
        self.alass_path = settings.ALASS_PATH
        self.timeout = settings.ALASS_TIMEOUT_SECONDS

    async def sync_async(self, target_bytes: bytes, reference_bytes: bytes) -> bytes | None:
        """Run alass alignment with bounded subprocess, timeout, and cleanup.

        The shared process-wide gate (``_ALASS_GATE``) ensures at most
        ``ALASS_MAX_CONCURRENCY`` subprocesses run at once across separate
        event‑loop / instance boundaries.  Cancellation cooperatively stops the
        worker and drains the gate permit on release.
        """
        stop = asyncio.Event()
        worker = asyncio.create_task(self._run(target_bytes, reference_bytes, stop))
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError:
            stop.set()
            while not worker.done():
                try:
                    await asyncio.shield(worker)
                except asyncio.CancelledError:
                    continue
            if not worker.cancelled():
                worker.exception()
            raise

    async def _run(self, target: bytes, reference: bytes, stop: asyncio.Event) -> bytes | None:
        """Core alass execution wrapped by cancellation / timeout / gate logic."""
        try:
            # Validate that the configured binary path is absolute and timeout is positive.
            if not os.path.isabs(self.alass_path) or not math.isfinite(self.timeout) or self.timeout <= 0:
                return None

            # Normalize both inputs (handles ASS/SSA → SRT, WebVTT → SRT, null/BOM stripping).
            target, target_cues = _normalize(target)
            reference, reference_cues = _normalize(reference)

            # Reject if the two timelines are not duration‑compatible.
            if not _compatible(target_cues, reference_cues):
                return None

            # Private temp directory; filenames are not derived from release names.
            with tempfile.TemporaryDirectory(prefix="ninjasubs-alass-") as directory:
                root = Path(directory)
                ref_path = root / "reference.srt"
                tgt_path = root / "target.srt"
                out_path = root / "output.srt"

                # Write input files; any I/O error aborts cleanly.
                try:
                    ref_path.write_bytes(reference)
                    tgt_path.write_bytes(target)
                except OSError:
                    return None

                # Acquire the process-wide gate before spawning; release in finally.
                # Non‑blocking acquire so a hung gate does not block the event loop.
                while not stop.is_set():
                    if _ALASS_GATE.acquire(blocking=False):
                        break
                    await asyncio.sleep(0.01)
                else:
                    return None
                try:
                    if stop.is_set():
                        return None

                    proc = await asyncio.create_subprocess_exec(
                        self.alass_path,
                        str(ref_path),
                        str(tgt_path),
                        str(out_path),
                        "--split-penalty",
                        "0.5",
                        stdin=asyncio.subprocess.DEVNULL,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        limit=8192,
                    )
                    assert proc.stdout is not None and proc.stderr is not None

                    readers = [
                        asyncio.create_task(_drain(proc.stdout, stop)),
                        asyncio.create_task(_drain(proc.stderr, stop)),
                    ]
                    complete = asyncio.gather(proc.wait(), *readers)
                    aborted = asyncio.create_task(stop.wait())

                    try:
                        done, _ = await asyncio.wait(
                            (complete, aborted),
                            timeout=self.timeout,
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        if complete not in done or stop.is_set():
                            return None
                        await complete
                        if proc.returncode != 0:
                            return None
                    finally:
                        await _reap(proc)
                        await asyncio.gather(complete, return_exceptions=True)
                        aborted.cancel()
                        await asyncio.gather(aborted, return_exceptions=True)
                finally:
                    _ALASS_GATE.release()

                # Validate the output file using strict OS‑level checks.
                try:
                    output = _read_output(out_path)
                except ValueError:
                    return None

                # Parse output cues and enforce retention / duration compatibility.
                try:
                    output_cues = _parse_srt(output.decode("utf-8"))
                except ValueError:
                    return None

                if len(output_cues) < math.ceil(len(target_cues) * 0.85):
                    return None
                if not _compatible(output_cues, target_cues):
                    return None

                return output
        except Exception:
            # No raw process output, exception paths or subtitle data appear in logs.
            return None
