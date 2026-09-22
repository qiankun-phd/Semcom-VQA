import unittest

from build_report import holm, paired_stats


class ReportStatisticsTests(unittest.TestCase):
    def test_identical_pairs(self):
        value = paired_stats([1, 0, 1], [1, 0, 1])
        self.assertEqual(value["mcnemar_exact_p"], 1.)
        self.assertEqual(value["ci95"], [0., 0.])

    def test_direction_and_exact_p(self):
        value = paired_stats([1] * 6, [0] * 6)
        self.assertEqual(value["gained"], 6)
        self.assertEqual(value["lost"], 0)
        self.assertEqual(value["mcnemar_exact_p"], .03125)
        self.assertEqual(value["difference"], 1.)

    def test_holm_is_monotonic(self):
        family = {"one": {"mcnemar_exact_p": .02}, "two": {"mcnemar_exact_p": .03}, "three": {"mcnemar_exact_p": .4}}
        holm(family)
        self.assertEqual(family["one"]["holm_adjusted_p"], .06)
        self.assertEqual(family["two"]["holm_adjusted_p"], .06)
        self.assertEqual(family["three"]["holm_adjusted_p"], .4)

    def test_invalid_pairing(self):
        with self.assertRaises(ValueError):
            paired_stats([], [])
        with self.assertRaises(ValueError):
            paired_stats([1], [1, 0])


if __name__ == "__main__":
    unittest.main()
