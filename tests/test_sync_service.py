"""Behavioral coverage for the bounded Alass runner."""
import asyncio
import os
from pathlib import Path

import pytest

from app.config import settings
from app.extractor import MAX_SUBTITLE_ENTRY_BYTES
from app.services import sync_service as module
from app.services.sync_service import SubtitleSyncService

pytestmark = pytest.mark.asyncio


def srt(count=10, text="مرحبا $(touch injected); & |", step=2):
    return "\n\n".join(
        f"{i}\n00:{(i * step) // 60:02d}:{(i * step) % 60:02d},000 --> "
        f"00:{(i * step + 1) // 60:02d}:{(i * step + 1) % 60:02d},000\n{text}"
        for i in range(1, count + 1)
    ).encode() + b"\n"


TARGET = srt()
REFERENCE = srt(text="English reference")


class Process:
    def __init__(self, harness):
        self.harness = harness
        self.returncode = None
        self.stdout = asyncio.StreamReader()
        self.stderr = asyncio.StreamReader()
        self.stdout.feed_data(harness.stdout)
        self.stderr.feed_data(harness.stderr)
        self.exited = asyncio.Event()
        self.terminated = self.killed = 0
        harness.active += 1
        harness.peak = max(harness.peak, harness.active)
        if not harness.hang:
            asyncio.get_running_loop().call_later(0.02, self.finish, harness.code)

    def finish(self, code):
        if self.returncode is None:
            self.returncode = code
            self.harness.active -= 1
            self.stdout.feed_eof()
            self.stderr.feed_eof()
            self.exited.set()

    async def wait(self):
        await self.exited.wait()
        return self.returncode

    def terminate(self):
        self.terminated += 1
        if not self.harness.ignore_term:
            self.finish(-15)

    def kill(self):
        self.killed += 1
        self.finish(-9)


class Harness:
    def __init__(self):
        self.calls = []
        self.processes = []
        self.directories = []
        self.inputs = []
        self.stdout = self.stderr = b""
        self.hang = self.ignore_term = False
        self.code = 0
        self.output = TARGET
        self.output_type = "regular"
        self.error = None
        self.started = asyncio.Event()
        self.spawn_release = None
        self.active = self.peak = 0

    async def spawn(self, *argv, **kwargs):
        self.calls.append((argv, kwargs))
        root = Path(argv[1]).parent
        self.directories.append(root)
        assert root.stat().st_mode & 0o077 == 0
        self.inputs.append((Path(argv[1]).read_bytes(), Path(argv[2]).read_bytes()))
        self.started.set()
        if self.spawn_release is not None:
            await self.spawn_release.wait()
        if self.error:
            raise self.error
        output = Path(argv[3])
        if self.output_type == "symlink":
            output.symlink_to(argv[2])
        elif self.output_type == "fifo":
            os.mkfifo(output)
        elif self.output_type == "directory":
            output.mkdir()
        elif self.output_type != "missing":
            output.write_bytes(self.output)
        proc = Process(self)
        self.processes.append(proc)
        return proc

    def assert_clean(self):
        assert self.active == 0
        assert all(not p.exists() for p in self.directories)
        assert all(p.returncode is not None for p in self.processes)


@pytest.fixture
def harness(monkeypatch):
    result = Harness()
    monkeypatch.setattr(module.asyncio, "create_subprocess_exec", result.spawn)
    monkeypatch.setattr(module, "TERMINATE_GRACE_SECONDS", 0.01)
    monkeypatch.setattr(settings, "ALASS_TIMEOUT_SECONDS", 0.2)
    yield result
    result.assert_clean()


async def test_success_argv_and_no_content_logging(harness, caplog):
    assert await SubtitleSyncService().sync_async(TARGET, REFERENCE) == TARGET
    argv, kwargs = harness.calls[0]
    assert argv[0] == "/usr/local/bin/alass"
    assert [Path(p).name for p in argv[1:4]] == ["reference.srt", "target.srt", "output.srt"]
    assert argv[4:] == ("--split-penalty", "0.5")
    assert len({Path(p).parent for p in argv[1:4]}) == 1
    assert kwargs.get("shell", False) is False
    assert harness.inputs == [(REFERENCE, TARGET)]
    assert "$(" not in repr(argv)
    assert "مرحبا" not in caplog.text
    assert "English reference" not in caplog.text


@pytest.mark.parametrize("which", ["target", "reference"])
@pytest.mark.parametrize("bad", [b"", b"bad", b"\xff", b"x" * (MAX_SUBTITLE_ENTRY_BYTES + 1)])
async def test_invalid_inputs_do_not_spawn(harness, which, bad):
    target, reference = (bad, REFERENCE) if which == "target" else (TARGET, bad)
    assert await SubtitleSyncService().sync_async(target, reference) is None
    assert not harness.calls


@pytest.mark.parametrize("path", ["alass", "./alass", ""])
async def test_relative_executable_rejected(harness, monkeypatch, path):
    monkeypatch.setattr(settings, "ALASS_PATH", path)
    assert await SubtitleSyncService().sync_async(TARGET, REFERENCE) is None
    assert not harness.calls


@pytest.mark.parametrize("error", [FileNotFoundError(), PermissionError(), OSError("private-data")])
async def test_spawn_failure_releases_permit(harness, error, caplog):
    harness.error = error
    assert await SubtitleSyncService().sync_async(TARGET, REFERENCE) is None
    harness.error = None
    assert await SubtitleSyncService().sync_async(TARGET, REFERENCE) == TARGET
    assert "private-data" not in caplog.text


@pytest.mark.parametrize("code", [1, 2, -9])
async def test_nonzero_never_accepts_output(harness, code):
    harness.code = code
    assert await SubtitleSyncService().sync_async(TARGET, REFERENCE) is None


@pytest.mark.parametrize("output", [
    b"", b"bad", b"\xff", b"x" * (MAX_SUBTITLE_ENTRY_BYTES + 1),
    TARGET.replace(b"00:00:02,000", b"-0:00:02,000"),
    TARGET.replace(b"00:00:02,000", b"00:00:62,000"),
    TARGET.replace(b"00:00:02,000", b"00:00:03,000"),
    srt(count=8), srt(step=8),
])
async def test_invalid_output_rejected(harness, output):
    harness.output = output
    assert await SubtitleSyncService().sync_async(TARGET, REFERENCE) is None


@pytest.mark.parametrize("kind", ["missing", "symlink", "fifo", "directory"])
async def test_output_file_safety(harness, kind):
    harness.output_type = kind
    assert await asyncio.wait_for(SubtitleSyncService().sync_async(TARGET, REFERENCE), 1) is None


async def test_gross_input_mismatch_rejected_before_spawn(harness):
    assert await SubtitleSyncService().sync_async(TARGET, srt(step=10)) is None
    assert not harness.calls


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
@pytest.mark.parametrize("size", [65536, 65537, 200000])
async def test_stream_limits(harness, stream, size, caplog):
    setattr(harness, stream, b"x" * size)
    result = await SubtitleSyncService().sync_async(TARGET, REFERENCE)
    assert result == (TARGET if size == 65536 else None)
    assert not caplog.text
    if size > 65536:
        assert harness.processes[0].terminated == 1


async def test_timeout_kills_then_releases_permit(harness, monkeypatch):
    harness.hang = harness.ignore_term = True
    monkeypatch.setattr(settings, "ALASS_TIMEOUT_SECONDS", 0.03)
    assert await SubtitleSyncService().sync_async(TARGET, REFERENCE) is None
    assert harness.processes[0].terminated == harness.processes[0].killed == 1
    harness.hang = harness.ignore_term = False
    monkeypatch.setattr(settings, "ALASS_TIMEOUT_SECONDS", 0.2)
    assert await SubtitleSyncService().sync_async(TARGET, REFERENCE) == TARGET


async def test_repeated_cancellation_reaps_and_releases(harness):
    harness.hang = harness.ignore_term = True
    task = asyncio.create_task(SubtitleSyncService().sync_async(TARGET, REFERENCE))
    await harness.started.wait()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert harness.processes[0].killed == 1
    harness.assert_clean()
    harness.hang = harness.ignore_term = False
    assert await SubtitleSyncService().sync_async(TARGET, REFERENCE) == TARGET


async def test_cancellation_during_spawn(harness):
    harness.spawn_release = asyncio.Event()
    harness.hang = True
    task = asyncio.create_task(SubtitleSyncService().sync_async(TARGET, REFERENCE))
    await harness.started.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    harness.spawn_release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert harness.processes[0].terminated == 1


async def test_shared_gate_across_instances(harness):
    first, second = SubtitleSyncService(), SubtitleSyncService()
    results = await asyncio.gather(first.sync_async(TARGET, REFERENCE), second.sync_async(TARGET, REFERENCE))
    assert results == [TARGET, TARGET]
    assert harness.peak == 1
    assert len(harness.calls) == 2


@pytest.mark.parametrize("fmt", ["srt", "ass", "ssa", "vtt"])
async def test_formats_normalize_to_srt(harness, fmt):
    if fmt == "srt":
        data = TARGET
    elif fmt == "vtt":
        data = b"WEBVTT\n\n" + TARGET.replace(b",000", b".000")
    else:
        field = "Layer" if fmt == "ass" else "Marked"
        data = (
            f"[Script Info]\nScriptType: v4.00+\n[Events]\nFormat: {field}, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
            + "\n".join(f"Dialogue: 0,0:00:{i*2:02}.00,0:00:{i*2+1:02}.00,Default,,0,0,0,,مرحبا" for i in range(1, 11))
        ).encode()
    assert await SubtitleSyncService().sync_async(data, REFERENCE) == TARGET
    normalized = harness.inputs[0][1]
    assert len(module._parse_srt(normalized.decode())) == 10
    assert b"Dialogue:" not in normalized and b"WEBVTT" not in normalized
