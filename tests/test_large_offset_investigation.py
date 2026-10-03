from __future__ import annotations

from pathlib import Path

import pytest

from app.services.sync.large_offset_investigation import (
    INV_REASON_LOW_TRUST,
    LargeOffsetDecision,
    investigate_large_offset,
)

FIX = Path(__file__).parent / "fixtures" / "large_offset"


def _read(name: str) -> str:
    return (FIX / name).read_text(encoding="utf-8")


@pytest.mark.parametrize("trust_value", ["invalid", "low", "", "HIGH"])
def test_unexpected_reference_trust_fails_closed(trust_value):
    inv = investigate_large_offset(
        _read("dexter_s08e04_target.srt"),
        _read("dexter_s08e04_reference.srt"),
        reference_trust=trust_value,
        identity_supported=True,
    )
    assert inv.decision == LargeOffsetDecision.INSUFFICIENT_EVIDENCE
    assert INV_REASON_LOW_TRUST in inv.reason_codes
