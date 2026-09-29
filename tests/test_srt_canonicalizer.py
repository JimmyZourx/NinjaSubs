"""Focused regression tests for the SRT canonicalization layer."""
from __future__ import annotations

import pytest

from app.services.sync_service import (
    _canonicalize_srt,
    _normalize,
    _parse_srt,
)


def test_ordinal_cue_ids_accepted():
    text = (
        "10th\n00:00:01,000 --> 00:00:02,000\nHello world\n\n"
        "11th\n00:00:03,000 --> 00:00:04,000\nSecond cue\n\n"
        "31th\n00:00:05,000 --> 00:00:06,000\nThird cue\n"
    )
    out = _canonicalize_srt(text)
    cues = _parse_srt(out)
    assert len(cues) == 3
    assert cues[0] == (1000, 2000, "Hello world")
    assert cues[1] == (3000, 4000, "Second cue")
    assert cues[2] == (5000, 6000, "Third cue")


def test_blank_line_before_cue_number():
    text = "\n\n1\n00:00:01,000 --> 00:00:02,000\nFirst\n\n2\n00:00:03,000 --> 00:00:04,000\nSecond\n"
    out = _canonicalize_srt(text)
    cues = _parse_srt(out)
    assert len(cues) == 2
    assert cues[0][2] == "First"
    assert cues[1][2] == "Second"


def test_malformed_arrow_dash_gt():
    text = "1\n00:00:01,000 -> 00:00:02,000\nA\n"
    out = _canonicalize_srt(text)
    cues = _parse_srt(out)
    assert len(cues) == 1
    assert cues[0][0] == 1000
    assert cues[0][1] == 2000


def test_malformed_arrow_dash_gt_dash():
    text = "1\n00:00:01,000 ->- 00:00:02,000\nA\n"
    out = _canonicalize_srt(text)
    cues = _parse_srt(out)
    assert len(cues) == 1
    assert cues[0][0] == 1000
    assert cues[0][1] == 2000


def test_malformed_arrow_dash_dash_gt_dash():
    text = "1\n00:00:01,000 -->- 00:00:02,000\nA\n"
    out = _canonicalize_srt(text)
    cues = _parse_srt(out)
    assert len(cues) == 1


def test_trailing_whitespace_on_timestamp_line():
    text = "1\n00:00:01,000 --> 00:00:02,000   \nBody line\n"
    out = _canonicalize_srt(text)
    cues = _parse_srt(out)
    assert cues[0][2] == "Body line"


def test_one_digit_milliseconds():
    text = "1\n00:00:01,7 --> 00:00:02,5\nSeven hundred ms\n"
    out = _canonicalize_srt(text)
    cues = _parse_srt(out)
    assert cues[0][0] == 1700
    assert cues[0][1] == 2500


def test_two_digit_milliseconds():
    text = "1\n00:00:01,72 --> 00:00:02,72\nSeventy two ms\n"
    out = _canonicalize_srt(text)
    cues = _parse_srt(out)
    assert cues[0][0] == 1720
    assert cues[0][1] == 2720


def test_three_digit_milliseconds_leading_zero():
    text = "1\n00:00:01,072 --> 00:00:02,072\nSeventy two ms\n"
    out = _canonicalize_srt(text)
    cues = _parse_srt(out)
    assert cues[0][0] == 1072
    assert cues[0][1] == 2072


def test_period_separator():
    text = "1\n00:00:01.000 --> 00:00:02.500\nPeriod ms\n"
    out = _canonicalize_srt(text)
    cues = _parse_srt(out)
    assert cues[0][0] == 1000
    assert cues[0][1] == 2500


def test_sequential_renumbering():
    text = (
        "5\n00:00:01,000 --> 00:00:02,000\nA\n\n"
        "42\n00:00:03,000 --> 00:00:04,000\nB\n\n"
        "7\n00:00:05,000 --> 00:00:06,000\nC\n"
    )
    out = _canonicalize_srt(text)
    cues = _parse_srt(out)
    assert len(cues) == 3
    assert cues[0][0] == 1000
    assert cues[1][0] == 3000
    assert cues[2][0] == 5000


def test_body_text_unchanged():
    body = "مرحبا $(touch injected); & | <font color=\"#ff0000\">red</font>\nline two"
    text = f"10th\n00:00:01,000 --> 00:00:02,000\n{body}\n"
    out = _canonicalize_srt(text)
    cues = _parse_srt(out)
    assert cues[0][2] == body


def test_malformed_timestamp_rejected():
    text = "1\n00:00:61,000 --> 00:00:02,000\nBad seconds\n"
    with pytest.raises(ValueError):
        _canonicalize_srt(text)


def test_no_timestamp_rejected():
    text = "1\nnot a timestamp\nbody\n"
    with pytest.raises(ValueError):
        _canonicalize_srt(text)


def test_empty_body_rejected():
    text = "1\n00:00:01,000 --> 00:00:02,000\n   \n"
    with pytest.raises(ValueError):
        _canonicalize_srt(text)


def test_start_ge_end_rejected():
    text = "1\n00:00:05,000 --> 00:00:02,000\nReversed\n"
    with pytest.raises(ValueError):
        _canonicalize_srt(text)


def test_nul_rejected():
    text = "1\n00:00:01,000 --> 00:00:02,000\nBad\x00byte\n"
    with pytest.raises(ValueError):
        _canonicalize_srt(text)


def test_valid_srt_unchanged_semantically():
    text = (
        "1\n00:00:01,000 --> 00:00:02,000\nFirst\n\n"
        "2\n00:00:03,000 --> 00:00:04,000\nSecond\n"
    )
    out = _canonicalize_srt(text)
    assert _parse_srt(out) == _parse_srt(text)


def test_regression_runtime_failure_combination():
    """Ordinal IDs + structural blank lines + malformed arrows + short milliseconds."""
    text = (
        "\n\n10th\n00:05:21,7 --> 00:05:23,0   \nHello world\n\n"
        "11th\n00:05:24,72 ->- 00:05:26,5\nSecond cue\n\n"
        "31th\n00:05:27.072 --> 00:05:29,000\nThird cue\n"
    )
    out = _canonicalize_srt(text)
    cues = _parse_srt(out)
    assert len(cues) == 3
    assert cues[0] == (321700, 323000, "Hello world")
    assert cues[1] == (324720, 326500, "Second cue")
    assert cues[2] == (327072, 329000, "Third cue")


def test_normalize_preserves_valid_srt_bytes():
    data = (
        b"1\n00:00:01,000 --> 00:00:02,000\nFirst\n\n"
        b"2\n00:00:03,000 --> 00:00:04,000\nSecond\n"
    )
    normalized, cues = _normalize(data)
    assert normalized == data
    assert len(cues) == 2


def test_normalize_rejects_garbage():
    with pytest.raises(ValueError):
        _normalize(b"garbage no subtitles here")


def test_normalize_rejects_nul_bytes():
    with pytest.raises(ValueError):
        _normalize(b"1\n00:00:01,000 --> 00:00:02,000\nBad\x00byte\n")
