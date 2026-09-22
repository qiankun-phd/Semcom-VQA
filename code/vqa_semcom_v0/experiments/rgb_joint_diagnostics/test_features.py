import unittest

import numpy as np

from feature_diagnostics import describe, distribution, question_audit


class FeatureDiagnosisTests(unittest.TestCase):
    def test_scale_and_power_share(self):
        rows = [{"question_features": [1.] + [0.] * 255, "image_features": [1.] * 83}]
        output = describe(rows, {"mean": [0.] * 83, "std": [1.] * 83})
        self.assertAlmostEqual(output["image_power_share"]["mean"], 83 / 84)
        self.assertAlmostEqual(output["image_over_question_l2"]["mean"], np.sqrt(83))

    def test_reject_empty_question(self):
        rows = [{"question_features": [0.] * 256, "image_features": [1.] * 83}]
        with self.assertRaises(ValueError):
            describe(rows, {"mean": [0.] * 83, "std": [1.] * 83})

    def test_nonfinite_distribution(self):
        with self.assertRaises(ValueError):
            distribution(np.asarray([float("nan")]))

    def test_overlap_not_truth_dependent(self):
        rows = [{"question": "Is it RED?", "question_type": "color", "answer": "ignored"}]
        output = question_audit(rows, {"is it red?"})
        self.assertEqual(output["questions_exactly_seen_in_train"], 1)


if __name__ == "__main__":
    unittest.main()
