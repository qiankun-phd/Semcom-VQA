import copy
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import revision_20260907_crossreceiver as cr


class CrossReceiverTests(unittest.TestCase):
    def sample(self):
        return {"image": "a", "question": "How many cars?", "snr": 0, "qt": "counting",
                "class": "car", "split": "train", "gt": 3, "answer": "3", "raw": 3,
                "1": {"raw": 3, "transmitted": 3, "correct": True, "calibrated": 3,
                      "prediction": "3", "normalized": "3", "evidence": "same packet"}}

    def test_only_old_count_calibration_may_differ(self):
        a = self.sample()
        b = copy.deepcopy(a)
        b["1"].update(correct=False, calibrated=8, prediction="8", normalized="8")
        self.assertEqual(sum(cr.compare_sender(a, b).values()), 4)
        b["1"]["transmitted"] = 4
        with self.assertRaises(ValueError):
            cr.compare_sender(a, b)

    def test_non_counting_label_difference_rejected(self):
        a = self.sample()
        a["qt"] = "presence"
        b = copy.deepcopy(a)
        b["1"]["correct"] = False
        with self.assertRaises(ValueError):
            cr.compare_sender(a, b)

    def test_gt_and_evidence_difference_rejected(self):
        a = self.sample()
        for field in ("gt", "answer"):
            b = copy.deepcopy(a)
            b[field] = "different"
            with self.assertRaises(ValueError):
                cr.compare_sender(a, b)
        b = copy.deepcopy(a)
        b["1"]["evidence"] = "different packet"
        with self.assertRaises(ValueError):
            cr.compare_sender(a, b)

    def test_coverage_preserves_unequal_task_mix(self):
        a = self.sample()
        b = copy.deepcopy(a)
        b.update(image="b", qt="presence")
        cov = cr.coverage([a, b])
        self.assertEqual(cov["train"]["decisions"], 2)
        self.assertEqual(cov["train"]["per_type"]["comparison"]["images"], 0)

    def test_summary_has_no_unmeasured_energy(self):
        a = self.sample()
        result = cr.summarize([a], np.array([[1, 0]]), np.array([0]), ["a"])
        self.assertEqual(result["pooled"], {"n": 1, "accuracy": 1., "image_fraction": 0.})

    def test_safe_features_ignore_annotations(self):
        a = self.sample()
        before = cr.base.features(a)
        a.update(gt=999, answer="no", polarity="negative", risk_level="critical")
        self.assertEqual(before, cr.base.features(a))
        self.assertEqual(len(before), 18)

    def test_restore_sender_preserves_original_log(self):
        a = self.sample()
        a["qt"] = "threshold"
        change = cr.restore_count(a, 4)
        self.assertEqual(change["logged_raw"], 3)
        self.assertEqual(a["raw"], 4)
        self.assertEqual(a["1"]["raw"], 3)
        self.assertEqual(a["1"]["transmitted"], 3)

    def test_sender_anchor_rejects_presence_counting_mismatch(self):
        with self.assertRaises(ValueError):
            cr.restore_count(self.sample(), 9)

    def test_restored_sender_invariant_across_snr(self):
        a = self.sample()
        a["qt"] = "comparison"
        b = copy.deepcopy(a)
        b.update(snr=-5, raw=1)
        cr.restore_count(a, 3)
        cr.restore_count(b, 3)
        self.assertEqual(cr.base.features(a)[-2:], cr.base.features(b)[-2:])


if __name__ == "__main__":
    unittest.main()
