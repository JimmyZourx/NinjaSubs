"""Focused offline tests; no model downloads or production imports."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
from semantic_guard import Cue, anchors_from_scores, guard, parse, render, robust_fit


class GuardTests(unittest.TestCase):
    def fixture(self):
        original = [Cue(str(i + 1), 100 + i * 60, 102 + i * 60, f"text {i}")
                    for i in range(30)]
        reference = [replace(c, start=c.start * 1.04 - 10, end=c.end * 1.04 - 10)
                     for c in original]
        anchors = [(i, i, 0.95, 0.3) for i in range(30)]
        return original, reference, list(reference), anchors

    def test_discovers_outlier_without_cue_ids(self):
        for index in (3, 14, 25):
            ar, ref, alass, anchors = self.fixture()
            alass[index] = replace(alass[index], start=alass[index].start + 25,
                                   end=alass[index].end + 25)
            output, report = guard(ar, ref, alass, anchors)
            self.assertEqual(report["corrected"], [index + 1])
            self.assertAlmostEqual(output[index].start, ref[index].start)
            for i in range(len(ar)):
                self.assertEqual(output[i].body, alass[i].body)
                if i != index:
                    self.assertEqual(output[i], alass[i])

    def test_low_confidence_and_unsupported_cues_unchanged(self):
        ar, ref, alass, anchors = self.fixture()
        alass[4] = replace(alass[4], start=400, end=402)
        output, report = guard(ar, ref, alass, anchors[:10])
        self.assertEqual(output, alass)
        self.assertFalse(report["confident"])
        output, report = guard(ar, ref, alass, [a for a in anchors if a[0] != 4])
        self.assertTrue(report["confident"])
        self.assertEqual(output, alass)
        self.assertEqual(len(report["suspicious"]), 1)

    def test_observe_only_reports_without_changing_timing(self):
        ar, ref, alass, anchors = self.fixture()
        alass[14] = replace(alass[14], start=alass[14].start + 25, end=alass[14].end + 25)
        output, report = guard(ar, ref, alass, anchors, observe_only=True)
        self.assertEqual(output, alass)
        self.assertEqual(report["mode"], "observe-only")
        self.assertEqual(report["corrected"], [])
        self.assertEqual(report["proposed_corrections"], [15])
        self.assertTrue(report["suspicious"][0]["proposed"])
        self.assertFalse(report["suspicious"][0]["corrected"])

    def test_nonlinear_timing_fails_confidence(self):
        ar, ref, alass, anchors = self.fixture()
        ref = [replace(c, start=c.start + (i % 3) * 10) for i, c in enumerate(ref)]
        output, report = guard(ar, ref, alass, anchors)
        self.assertFalse(report["confident"])
        self.assertEqual(output, alass)

    def test_mismatched_text_or_count_rejected(self):
        ar, ref, alass, anchors = self.fixture()
        with self.assertRaises(ValueError):
            guard(ar, ref, alass[:-1], anchors)
        alass[5] = replace(alass[5], body="different")
        with self.assertRaises(ValueError):
            guard(ar, ref, alass, anchors)

    def test_robust_fit_rejects_timing_outlier(self):
        x = np.arange(30) * 60.0
        y = x * 1.04 - 10
        y[12] += 100
        slope, offset, mask = robust_fit(x, y)
        self.assertAlmostEqual(slope, 1.04)
        self.assertAlmostEqual(offset, -10)
        self.assertFalse(mask[12])

    def test_ambiguous_embeddings_not_trusted(self):
        self.assertEqual(anchors_from_scores(np.ones((3, 3))), [])
        self.assertEqual(len(anchors_from_scores(np.eye(5))), 5)

    def test_flexible_srt_and_invalid_duration(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / "sample.srt"
            path.write_text("10th\n00:00:01.5 -> 00:00:02,72\n<i>Hello</i>\n\n\n"
                            "11th\n00:00:03,000 --> 00:00:04,000\nWorld\n", encoding="utf-8")
            cues = parse(path)
            self.assertEqual(len(cues), 2)
            self.assertEqual(cues[0].start, 1.5)
            self.assertAlmostEqual(cues[0].end, 2.72)
            path.write_text(render(cues), encoding="utf-8")
            self.assertEqual(parse(path), cues)
            path.write_text("1\n00:00:02,000 --> 00:00:01,000\nBad", encoding="utf-8")
            with self.assertRaises(ValueError):
                parse(path)


if __name__ == "__main__":
    unittest.main()
