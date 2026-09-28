"""Stage 2 reference decoder tests; ZIP handling is delegated to app.extractor."""

import io
import zipfile
from unittest.mock import MagicMock, patch

import pytest

from app.extractor import (
    MAX_SUBTITLE_ENTRY_BYTES,
    MAX_TOTAL_UNCOMPRESSED_BYTES,
    MAX_ZIP_COMPRESSION_RATIO,
    MAX_ZIP_INPUT_BYTES,
    SubtitleExtractionError,
    extract_srt_from_zip,
)
from app.services.sync.decode import decode_payload


def _zip(entries: list[tuple[str, bytes]], *, compression=zipfile.ZIP_DEFLATED) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=compression) as archive:
        for name, value in entries:
            archive.writestr(name, value)
    return output.getvalue()


def _reference(text: str) -> bytes:
    return ("1\n00:00:01,000 --> 00:00:02,000\n" + text + "\n").encode()


@pytest.mark.parametrize(
    ("name", "payload", "marker"),
    [
        ("reference.srt", _reference("SRT reference"), b"SRT reference"),
        (
            "reference.ass",
            b"[Script Info]\n[V4+ Styles]\n[Events]\nDialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,ASS reference\n",
            b"ASS reference",
        ),
        (
            "reference.ssa",
            b"[Script Info]\n[V4 Styles]\n[Events]\nDialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,SSA reference\n",
            b"SSA reference",
        ),
        (
            "reference.vtt",
            b"WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nVTT reference\n",
            b"VTT reference",
        ),
    ],
)
def test_reference_decode_accepts_supported_secure_zip_formats(name, payload, marker):
    decoded = decode_payload(_zip([(name, payload)]), name, min_bytes=1)
    assert decoded is not None
    assert marker in decoded


def test_plain_subtitle_over_entry_limit_is_rejected_before_decode():
    assert decode_payload(b"1\n" * (MAX_SUBTITLE_ENTRY_BYTES // 2 + 1), "large.srt") is None


def test_zip_over_input_limit_is_rejected():
    assert decode_payload(b"PK" + b"x" * MAX_ZIP_INPUT_BYTES, "big.zip") is None


@pytest.mark.parametrize(
    "archive",
    [
        _zip([(f"junk-{n}.bin", b"x") for n in range(201)]),
        _zip([("large.srt", b"x" * (MAX_SUBTITLE_ENTRY_BYTES + 1))]),
        _zip([(f"part-{n}.srt", _reference("x" * (MAX_SUBTITLE_ENTRY_BYTES // 2))) for n in range(3)]),
        _zip([("bomb.srt", b"0\n" * 100_000)]),
    ],
)
def test_reference_decode_reuses_extractor_limits(archive):
    # The helper returns an unusable reference for each guarded archive.
    assert decode_payload(archive, "reference", min_bytes=1) is None


def test_zip_slip_and_long_path_behavior_comes_from_current_extractor():
    safe = _reference("safe subtitle")
    mixed = _zip([("../../escape.srt", _reference("bad")), ("nested/safe.srt", safe)])
    decoded = decode_payload(mixed, "safe", min_bytes=1)
    assert decoded is not None and b"safe subtitle" in decoded and b"bad" not in decoded

    long_member = _zip([("a" * 260 + ".srt", safe)])
    assert decode_payload(long_member, "safe", min_bytes=1) is None


def test_decode_wrapper_enforces_aggregate_uncompressed_limit_independently():
    class Entry:
        def __init__(self, index):
            self.filename = f"part-{index}.srt"
            self.file_size = 2_100_000
            self.compress_size = 2_100_000

        def is_dir(self):
            return False

    with patch("app.extractor.zipfile.ZipFile") as zip_file:
        archive = MagicMock()
        archive.infolist.return_value = [Entry(index) for index in range(10)]
        zip_file.return_value = archive
        assert decode_payload(b"PK" + b"x" * 20, "parts.zip", min_bytes=1) is None


def test_decode_wrapper_enforces_compression_ratio_limit():
    class Entry:
        filename = "bomb.srt"
        file_size = 300_000
        compress_size = 1_000

        @staticmethod
        def is_dir():
            return False

    with patch("app.extractor.zipfile.ZipFile") as zip_file:
        archive = MagicMock()
        archive.infolist.return_value = [Entry()]
        zip_file.return_value = archive
        assert decode_payload(b"PK" + b"x" * 20, "bomb.zip", min_bytes=1) is None


def test_invalid_archive_is_unusable_and_extractor_raises_clean_error():
    with pytest.raises(SubtitleExtractionError):
        extract_srt_from_zip(b"PK-not-a-valid-zip")
    assert decode_payload(b"PK-not-a-valid-zip", "bad.zip") is None


def test_extractor_limit_constants_are_the_stage2_policy():
    assert MAX_ZIP_INPUT_BYTES == 10 * 1024 * 1024
    assert MAX_SUBTITLE_ENTRY_BYTES == 5 * 1024 * 1024
    assert MAX_TOTAL_UNCOMPRESSED_BYTES == 20 * 1024 * 1024
    assert MAX_ZIP_COMPRESSION_RATIO == 200
