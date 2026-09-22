import csv
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import revision_20260907 as base
import revision_20260907_dronevehicle as dv
import revision_20260907_supplement as supp


class DroneVehicleTests(unittest.TestCase):
    def fixture(self):
        rows = []
        for split in ("train", "validation", "test"):
            iid = next(str(i) for i in range(1000) if base.split_for(str(i)) == split)
            for branch in ("1", "2"):
                rows.append(dict(image_id=iid, question="Are there car objects?", question_type="presence",
                    target_class="car", snr_bin="0dB", service_level=branch, correct="True", payload_bytes="100",
                    raw_detector_count="4", transmitted_detector_count="4", calibrated_detector_count="4",
                    object_count="4", ground_truth_answer="yes", presence_polarity="positive", image_path="/tmp/image.jpg",
                    model_name="Qwen2-VL-2B-Instruct" if branch == "2" else "semantic-token-decoder"))
        return rows

    def load(self, rows):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "data.csv"
            with p.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=rows[0].keys())
                w.writeheader()
                w.writerows(rows)
            return dv.strict_groups(p)

    def test_metadata_conflict_rejected(self):
        rows = self.fixture()
        rows[1]["target_class"] = "truck"
        with self.assertRaisesRegex(ValueError, "metadata"):
            self.load(rows)

    def test_unknown_class_rejected(self):
        rows = self.fixture()
        for row in rows:
            row["target_class"] = "unknown"
        with self.assertRaisesRegex(ValueError, "Unknown class"):
            self.load(rows)

    def test_wrong_receiver_rejected(self):
        rows = self.fixture()
        rows[1]["model_name"] = "other"
        with self.assertRaisesRegex(ValueError, "receiver"):
            self.load(rows)

    def test_safe_features_ignore_answers(self):
        rows, _ = self.load(self.fixture())
        g = rows[0]
        x = base.features(g)
        g["1"]["answer"] = "no"
        g["1"]["polarity"] = "negative"
        self.assertEqual(x, base.features(g))
        self.assertEqual(len(x), 18)

    def test_no_target_calibration_for_transfer(self):
        groups, _ = self.load(self.fixture())
        for g in groups:
            g["qt"] = "counting"
            g["1"]["gt"] = 8
        np.testing.assert_array_equal(base.labels(groups, {"car|0": 2})[:, 0], 1)
        np.testing.assert_array_equal(base.labels(groups, {})[:, 0], 0)
        with self.assertRaises(ValueError):
            base.fit_calibration(groups)

    def test_selection_validation_only_and_ties(self):
        s = [{"lambda": 0., "accuracy": .7, "energy_j": 9},
             {"lambda": .01, "accuracy": .695, "energy_j": 3},
             {"lambda": .1, "accuracy": .68, "energy_j": 1}]
        self.assertEqual(supp.select_price(s), 1)

    def test_repair_receiver_count_with_original_boxes(self):
        rows = self.fixture()
        extras = []
        for row in rows:
            extra = dict(row, question_type="threshold", question="At least four cars?", raw_detector_count="3")
            extras.append(extra)
        rows += extras
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "outputs/detector").mkdir(parents=True)
            for tag in ("main", "cmp", "extra"):
                fields = ("image_id", "category", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "confidence", "detector_model")
                with (root / f"outputs/detector/dv_rician_{tag}_detections.csv").open("w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=fields)
                    w.writeheader()
                    for iid in {r["image_id"] for r in rows}:
                        for i in range(4):
                            w.writerow(dict(image_id=iid, category="car", bbox_x=i, bbox_y=0, bbox_w=1, bbox_h=1, confidence=.8, detector_model="frozen"))
            p = root / "predictions.csv"
            with p.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=rows[0].keys())
                w.writeheader()
                w.writerows(rows)
            groups, audit = dv.strict_groups(p, root)
            self.assertEqual(len(audit["logged_raw_field_corrections"]), 3)
            self.assertTrue(all(g["raw"] == 4 for g in groups))
            self.assertTrue(all(g["1"]["raw"] == 3 for g in groups if g["qt"] == "threshold"))


if __name__ == "__main__":
    unittest.main()
