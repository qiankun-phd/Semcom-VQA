"""Regression checks for frozen supplement selection and ablation accounting."""
from pathlib import Path
import sys
import unittest

import numpy as np

SOURCE = Path(__file__).parents[1] / "scripts"
if not (SOURCE / "revision_20260907_supplement.py").exists():
    SOURCE = Path(__file__).parent
sys.path.insert(0, str(SOURCE))
import revision_20260907_supplement as s


class SupplementTests(unittest.TestCase):
    def test_feature_indices(self):
        self.assertEqual(s.VARIANTS["no_detector"], list(range(16)))
        self.assertEqual(s.VARIANTS["no_snr"], list(range(15)) + [16, 17])
        self.assertEqual(s.VARIANTS["question_type_only"], list(range(5)))

    def test_selection_validation_threshold_and_tie(self):
        rows = [{"lambda": 0, "accuracy": .8, "energy_j": 10},
                {"lambda": .01, "accuracy": .795, "energy_j": 3},
                {"lambda": .02, "accuracy": .79, "energy_j": 3},
                {"lambda": .03, "accuracy": .78, "energy_j": 1}]
        self.assertEqual(s.select_price(rows), 1)
        # No test metrics are part of the interface.
        with self.assertRaises(TypeError):
            s.select_price(rows, test_accuracy=[0, 0, 1, 1])

    def test_all_grid_points_and_picks_saved(self):
        y = np.array([[0, 1], [1, 0]])
        p = np.array([[.3, .6], [.8, .2]])
        energy = np.array([[.43, 32.8], [.43, 32.8]])
        rows, picks = s.sweep(y, p, energy)
        self.assertEqual(len(rows), 26)
        self.assertEqual(picks.shape, (26, 2))
        self.assertEqual([r["lambda"] for r in rows], s.base.PRICES)
        for r, pk in zip(rows, picks):
            self.assertEqual(r["accuracy"], float(y[np.arange(2), pk].mean()))

    def test_reused_mismatch_stops(self):
        s.require_same({"hash": "x"}, {"hash": "x"}, "fixture")
        with self.assertRaises(ValueError):
            s.require_same({"hash": "x"}, {"hash": "y"}, "fixture")

    def test_detector_sensitivity_only_image_branch(self):
        energy = np.array([[.5, 33.], [.6, 34.]])
        for variant in ("no_detector", "question_type_only"):
            new = s.deployment_energy(energy, variant)
            np.testing.assert_array_equal(new[:, 0], energy[:, 0])
            np.testing.assert_allclose(energy[:, 1] - new[:, 1], s.base.ENERGY_DET)
        np.testing.assert_array_equal(s.deployment_energy(energy, "no_snr"), energy)
        np.testing.assert_array_equal(energy, [[.5, 33.], [.6, 34.]])


if __name__ == "__main__":
    unittest.main()
