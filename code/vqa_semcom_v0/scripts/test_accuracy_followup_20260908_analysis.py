import unittest
import numpy as np
from accuracy_followup_20260908_analyze_c import contrast, summarize


class AnalysisTests(unittest.TestCase):
    def test_zero_paired_difference(self):
        v = contrast(np.zeros(6), np.array(['a', 'a', 'b', 'b', 'c', 'c']))
        self.assertEqual(v['effect_pp'], 0)
        self.assertEqual(v['bootstrap_95_percentile_pp'], [0, 0])
        self.assertEqual(v['two_sided_sign_flip_p'], 1)

    def test_pooled_not_unweighted_image_average(self):
        v = contrast(np.array([1., 1., 1., 0.]), np.array(['a', 'a', 'a', 'b']))
        self.assertEqual(v['effect_pp'], 75)

    def test_actual_coverage(self):
        records = [{'image': 'a', 'correct': True, 'normalized': 'yes'},
                   {'image': 'a', 'correct': False, 'normalized': 'unknown'},
                   {'image': 'b', 'correct': False, 'normalized': 'no'}]
        v = summarize(records)
        self.assertEqual((v['n'], v['images'], v['correct'], v['unknown']), (3, 2, 1, 1))
        self.assertEqual(v['accuracy'], 1 / 3)


if __name__ == '__main__':
    unittest.main()
