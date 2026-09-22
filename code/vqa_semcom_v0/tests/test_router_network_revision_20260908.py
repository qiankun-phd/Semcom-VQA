import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import router_network_revision_20260908 as r


class RouterTests(unittest.TestCase):
    def test_target_signed_difference(self):
        y = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=np.int8)
        mask, target = r.targets(y, "advantage_mse", 0)
        np.testing.assert_array_equal(target, [-1, 1, 0, 0])
        self.assertTrue(mask.all())

    def test_disagreement_excludes_equal_outcomes(self):
        y = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=np.int8)
        mask, target = r.targets(y, "disagreement_bce", 0)
        np.testing.assert_array_equal(mask, [1, 1, 0, 0])
        np.testing.assert_array_equal(target, [0, 1])

    def test_route_tie_chooses_detection(self):
        self.assertEqual(r.accuracy(np.array([[1, 0], [0, 1]]), np.array([0., 0.])), .5)

    def test_selection_only_validation_not_extra_test_field(self):
        records = {k: [{"validation_accuracy": .7, "test_accuracy": i}] * 3 for i, k in enumerate(r.CONFIGS)}
        self.assertEqual(r.select(records), "advantage_mse")
        records["wide_bce"] = [{"validation_accuracy": .71, "test_accuracy": -100}] * 3
        self.assertEqual(r.select(records), "wide_bce")

    def test_parameter_counts(self):
        self.assertEqual(r.parameter_count(r.CONFIGS["original_bce"]), 2306)
        self.assertEqual(r.parameter_count(r.CONFIGS["wide_bce"]), 6658)
        self.assertEqual(r.parameter_count(r.CONFIGS["advantage_mse"]), 1153)


if __name__ == "__main__":
    unittest.main()
