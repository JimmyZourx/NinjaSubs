"""Regression tests for ZIP resource limits (P1-A)."""

import io
import zipfile
from unittest.mock import MagicMock, patch

import pytest

from app.extractor import (
    MAX_SUBTITLE_ENTRY_BYTES,
    MAX_ZIP_INPUT_BYTES,
    SubtitleExtractionError,
    extract_srt_from_zip,
    is_zip_slip_attempt,
)


def create_mock_zip(files_dict: dict) -> bytes:
    """Helper to create an in-memory ZIP archive from a dict of {filename: bytes_or_str}."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, content in files_dict.items():
            if isinstance(content, str):
                content = content.encode("utf-8")
            zf.writestr(name, content)
    return buf.getvalue()


def create_zip_with_entry(filename: str, size: int, compress: bool = True) -> bytes:
    """Create a ZIP with a single entry of specified size."""
    content = b"x" * size
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED) as zf:
        zf.writestr(filename, content)
    return buf.getvalue()


def test_extract_rejects_zip_over_10mb_input():
    """ZIP input >10 MB rejected before opening."""
    big = b"x" * (MAX_ZIP_INPUT_BYTES + 1)
    with pytest.raises(SubtitleExtractionError, match="ZIP input exceeds maximum size"):
        extract_srt_from_zip(big)


def test_extract_rejects_over_200_total_members():
    """Archive with >200 total members rejected before processing."""
    # 150 .srt + 50 .txt + 10 .nfo = 210 total members
    files = {}
    for i in range(150):
        files[f"sub_{i}.srt"] = f"Subtitle {i}"
    for i in range(50):
        files[f"junk_{i}.txt"] = "junk"
    for i in range(10):
        files[f"info_{i}.nfo"] = "info"
    zip_bytes = create_mock_zip(files)
    with pytest.raises(SubtitleExtractionError, match="Archive entry limit exceeded"):
        extract_srt_from_zip(zip_bytes)


def test_extract_rejects_over_200_non_subtitle_plus_one_srt():
    """Archive with >200 non-subtitle/junk members + 1 valid SRT rejected."""
    files = {"valid.srt": "Real subtitle"}
    for i in range(200):
        files[f"junk_{i}.dat"] = "junk"
    zip_bytes = create_mock_zip(files)
    with pytest.raises(SubtitleExtractionError, match="Archive entry limit exceeded"):
        extract_srt_from_zip(zip_bytes)


def test_extract_rejects_single_entry_over_5mb():
    """Single subtitle entry >5 MB rejected."""
    zip_bytes = create_zip_with_entry("movie.srt", MAX_SUBTITLE_ENTRY_BYTES + 1)
    with pytest.raises(SubtitleExtractionError, match="Entry exceeds maximum size"):
        extract_srt_from_zip(zip_bytes)


def test_extract_rejects_total_uncompressed_over_20mb_independently():
    """Total uncompressed >20 MB rejected independently from compression ratio.

    Uses mocked ZipInfo to control metadata, ensuring ratio stays ~1:1
    while total uncompressed exceeds 20MB.
    """

    zip_bytes = create_mock_zip({"movie.srt": "x" * 1000})

    class MockEntry:
        def __init__(self, name, compressed, uncompressed):
            self.filename = name
            self.compress_size = compressed
            self.file_size = uncompressed
            self.is_dir = lambda: False

    # 10 entries of 2.1MB each = 21MB total, ratio ~1:1
    entries = [MockEntry(f"sub_{i}.srt", 2_100_000, 2_100_000) for i in range(10)]

    # Mock the entire ZipFile class to control infolist() and read()
    with patch('app.extractor.zipfile.ZipFile') as mock_zipfile_class:
        mock_zf = MagicMock()
        mock_zf.infolist.return_value = entries
        mock_zf.read.return_value = b"x" * 2_100_000
        mock_zf.__enter__.return_value = mock_zf
        mock_zf.__exit__.return_value = None
        mock_zipfile_class.return_value = mock_zf

        with pytest.raises(SubtitleExtractionError, match="Total uncompressed size exceeds limit"):
            extract_srt_from_zip(zip_bytes)


def test_extract_rejects_compression_ratio_over_200():
    """Compression ratio >200:1 rejected (zip bomb defense).

    Uses mocked ZipInfo metadata since real compression can't easily
    produce 200:1 ratio. Patches ZipInfo metadata during iteration.
    """

    zip_bytes = create_mock_zip({"movie.srt": "x" * 1000})

    class MockEntry:
        def __init__(self, name, compressed, uncompressed):
            self.filename = name
            self.compress_size = compressed
            self.file_size = uncompressed
            self.is_dir = lambda: False

    # 10001:50 = 200:1 ratio, should trigger limit
    entry = MockEntry("movie.srt", 50, 10001)

    with patch('app.extractor.zipfile.ZipFile') as mock_zipfile_class:
        mock_zf = MagicMock()
        mock_zf.infolist.return_value = [entry]
        mock_zf.read.return_value = b"x" * 10001
        mock_zf.__enter__.return_value = mock_zf
        mock_zf.__exit__.return_value = None
        mock_zipfile_class.return_value = mock_zf

        with pytest.raises(SubtitleExtractionError, match="Compression ratio exceeds limit"):
            extract_srt_from_zip(zip_bytes)


def test_extract_rejects_zero_compressed_size_with_uncompressed():
    """Entry with compress_size == 0 and uncompressed > 0 rejected."""

    zip_bytes = create_mock_zip({"movie.srt": "x" * 1000})

    class MockEntry:
        def __init__(self, name, compressed, uncompressed):
            self.filename = name
            self.compress_size = compressed
            self.file_size = uncompressed
            self.is_dir = lambda: False

    entry = MockEntry("movie.srt", 0, 1000)

    with patch('app.extractor.zipfile.ZipFile') as mock_zipfile_class:
        mock_zf = MagicMock()
        mock_zf.infolist.return_value = [entry]
        mock_zf.__enter__.return_value = mock_zf
        mock_zf.__exit__.return_value = None
        mock_zipfile_class.return_value = mock_zf

        with pytest.raises(SubtitleExtractionError, match="Invalid compression size"):
            extract_srt_from_zip(zip_bytes)


def test_extract_rejects_path_over_256_chars():
    """Archive member path >256 chars rejected."""
    long_name = "a" * 260 + ".srt"
    zip_bytes = create_mock_zip({long_name: "subtitle"})
    with pytest.raises(SubtitleExtractionError, match="Archive path exceeds limit"):
        extract_srt_from_zip(zip_bytes)


def test_zip_slip_protection_still_works():
    """Zip Slip protection still blocks traversal attempts."""
    assert is_zip_slip_attempt("../../evil.srt") is True
    assert is_zip_slip_attempt("..\\..\\evil.srt") is True
    assert is_zip_slip_attempt("/etc/passwd.srt") is True
    assert is_zip_slip_attempt("C:\\Windows\\evil.srt") is True
    assert is_zip_slip_attempt("normal_folder/subtitle.srt") is False
    assert is_zip_slip_attempt("subtitle.srt") is False

    zip_bytes = create_mock_zip(
        {
            "../../etc/evil.srt": "Evil",
            "subtitles/safe_release.srt": "Safe",
        }
    )
    extracted = extract_srt_from_zip(zip_bytes)
    assert extracted.decode("utf-8") == "Safe"


def test_valid_normal_zip_still_extracts():
    """Valid normal ZIP still extracts correctly."""
    srt_content = "1\n00:00:01,000 --> 00:00:03,000\nValid subtitle\n"
    zip_bytes = create_mock_zip({"movie.srt": srt_content})
    extracted = extract_srt_from_zip(zip_bytes)
    assert extracted.decode("utf-8") == srt_content


def test_valid_ass_ssa_vtt_still_work():
    """ASS, SSA, VTT subtitles still extract correctly."""
    ass_content = (
        "[Script Info]\n"
        "Title: Test\n"
        "[V4+ Styles]\n"
        "Style: Default,Arial,20,&H00FFFFFF\n"
        "[Events]\n"
        "Dialogue: 0,0:00:01.00,0:00:04.00,Default,,0,0,0,,Test\n"
    )
    zip_bytes = create_mock_zip({"movie.ass": ass_content})
    extracted = extract_srt_from_zip(zip_bytes)
    assert "[Script Info]" in extracted.decode("utf-8")
    assert "Dialogue:" in extracted.decode("utf-8")

    vtt_content = "WEBVTT\n\n1\n00:00:01.000 --> 00:00:03.000\nTest\n"
    zip_bytes = create_mock_zip({"movie.vtt": vtt_content})
    extracted = extract_srt_from_zip(zip_bytes)
    assert "WEBVTT" in extracted.decode("utf-8")

    ssa_content = ass_content.replace("[V4+ Styles]", "[V4 Styles]")
    zip_bytes = create_mock_zip({"movie.ssa": ssa_content})
    extracted = extract_srt_from_zip(zip_bytes)
    assert "[V4 Styles]" in extracted.decode("utf-8")


def test_extract_still_selects_best_episode():
    """Episode selection logic still works with multiple entries."""
    ep1 = "Ep 1"
    ep2 = "Ep 2"
    ep3 = "Ep 3"
    zip_bytes = create_mock_zip(
        {
            "Show.S02E01.srt": ep1,
            "Show.S02E02.srt": ep2,
            "Show.S02E03.srt": ep3,
        }
    )
    extracted = extract_srt_from_zip(
        zip_bytes,
        target_filename="Show.S02E02.srt",
        season=2,
        episode=2,
    )
    assert extracted.decode("utf-8") == ep2


def test_macos_metadata_ignored():
    """__MACOSX and ._ files still ignored."""
    zip_bytes = create_mock_zip(
        {
            "__MACOSX/._movie.srt": b"metadata",
            "movie.srt": "Real subtitle",
        }
    )
    extracted = extract_srt_from_zip(zip_bytes)
    assert extracted.decode("utf-8") == "Real subtitle"


def test_directory_entries_ignored():
    """Directory entries still ignored."""
    zip_bytes = create_mock_zip(
        {
            "subtitles/": b"",
            "subtitles/movie.srt": "Subtitle",
        }
    )
    extracted = extract_srt_from_zip(zip_bytes)
    assert extracted.decode("utf-8") == "Subtitle"


def test_total_uncompressed_check_triggers_before_ratio():
    """Test that total uncompressed >20MB triggers before ratio check.

    Uses data that compresses well (so ratio is low) but total exceeds 20MB.
    """

    zip_bytes = create_mock_zip({"movie.srt": "x" * 1000})

    class MockEntry:
        def __init__(self, name, compressed, uncompressed):
            self.filename = name
            self.compress_size = compressed
            self.file_size = uncompressed
            self.is_dir = lambda: False

    # 180 entries of 120KB each = 21.6MB total, highly compressible (ratio ~1:120)
    # Stay under MAX_ARCHIVE_ENTRIES (200) so entry count check doesn't trigger
    entries = [MockEntry(f"sub_{i}.srt", 1000, 120_000) for i in range(180)]

    # Mock the entire ZipFile class to control infolist() and read()
    with patch('app.extractor.zipfile.ZipFile') as mock_zipfile_class:
        mock_zf = MagicMock()
        mock_zf.infolist.return_value = entries
        mock_zf.read.return_value = b"x" * 120_000
        mock_zf.__enter__.return_value = mock_zf
        mock_zf.__exit__.return_value = None
        mock_zipfile_class.return_value = mock_zf

        with pytest.raises(SubtitleExtractionError, match="Total uncompressed size exceeds limit"):
            extract_srt_from_zip(zip_bytes)
