"""Boundary and mapping tests for the frozen EXP-012 scoring diagnosis."""
from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_diagnostics import (CELLS, VARIANTS, action_summary, auc, choose_action,
                               image_mean_interval, nominal_cost, normalize,
                               penalty_decomposition, scoring_metrics)


def fixture(labels: list[int]) -> dict:
    return {"id": "example", "correct": labels, "actual_cost": [nominal_cost(a) for a in range(9)],
            "image_bytes": [budget + 1 for budget, _ in CELLS],
            "visual_tokens": [64 * (1, 2, 4)[a % 3] for a in range(9)]}


class ScoreBoundaries(unittest.TestCase):
    def test_auc_constant_labels_undefined(self) -> None:
        self.assertIsNone(auc([0] * 9, [.5] * 9))
        self.assertIsNone(auc([1] * 9, [.5] * 9))

    def test_auc_ties_receive_half_credit(self) -> None:
        self.assertEqual(auc([0, 1], [.4, .4]), .5)
        self.assertEqual(auc([0, 1, 1], [.2, .2, .8]), .75)
        self.assertEqual(auc([0, 0, 1, 1], [.1, .2, .8, .9]), 1.)
        self.assertEqual(auc([1, 1, 0, 0], [.1, .2, .8, .9]), 0.)

    def test_action_grid_and_noncontiguous_head_mapping(self) -> None:
        self.assertEqual(CELLS[3], (4000, "low"))
        self.assertEqual(CELLS[5], (4000, "high"))
        self.assertEqual(VARIANTS["rate_at_low"], [0, 3, 6])
        self.assertEqual(choose_action([.1, .2, .9], [0, 3, 6]), 6)
        self.assertEqual(choose_action([.9, .1, .2], [3, 4, 5]), 3)

    def test_probability_ties_choose_cheaper_action_for_both_lambdas(self) -> None:
        for weight in (0, .05):
            self.assertEqual(choose_action([.5, .5, .5], [6, 0, 3], weight), 0)
            self.assertEqual(choose_action([.5, .5, .5], [5, 4, 3], weight), 3)

    def test_only_prespecified_lambdas_and_valid_probabilities(self) -> None:
        with self.assertRaises(ValueError):
            choose_action([.5], [0], .01)
        for scores, actions in (([math.nan], [0]), ([1.1], [0]), ([.2], [9]), ([.2, .2], [0, 0]), ([.2], [0, 3])):
            with self.assertRaises(ValueError):
                choose_action(scores, actions)

    def test_all_correct_and_all_wrong_images_have_no_rescue_opportunity(self) -> None:
        for y in (0, 1):
            rows = [fixture([y] * 9)]
            result = scoring_metrics(rows, list(range(9)), [[.5] * 9], False)
            self.assertEqual(result["mixed_candidate_images"], 0)
            self.assertIsNone(result["within_image_macro_auc_mixed_only"])
            selected = result["selected_lambda005"]
            self.assertEqual(selected["eligible_low_wrong_any_other_correct"], 0)
            self.assertEqual(selected["rescued_vs_2000_low"], 0)
            self.assertEqual(selected["harmed_vs_2000_low"], 0)
            self.assertEqual(selected["correct"], y)

    def test_penalty_can_erase_or_repair_a_probability_rank_success(self) -> None:
        probabilities = [[.5, .51] + [.1] * 7] * 2
        rescue = fixture([0, 1] + [0] * 7)
        damage = fixture([1, 0] + [0] * 7)
        result = penalty_decomposition([rescue, damage], probabilities, list(range(9)))
        mixed = result["candidate_mixed_outcomes"]
        self.assertEqual(mixed["n"], 2)
        self.assertEqual(mixed["penalty_changed_correct_to_wrong"], 1)
        self.assertEqual(mixed["penalty_changed_wrong_to_correct"], 1)
        self.assertEqual(mixed["wrong_without_penalty_and_with_penalty"], 0)
        self.assertEqual(result["candidate_oracle_errors_without_penalty"], 1)
        self.assertEqual(result["candidate_oracle_errors_original"], 1)

    def test_rescue_is_conditional_on_low_wrong_and_any_other_correct(self) -> None:
        rows = [fixture([0, 0, 0, 1, 0, 0, 0, 0, 0]), fixture([0] * 9), fixture([1, 0, 0, 0, 0, 0, 0, 0, 0])]
        result = action_summary(rows, [0, 3, 6], [[.1, .9, .2]] * 3, 0)
        self.assertEqual(result["eligible_low_wrong_any_other_correct"], 1)
        self.assertEqual(result["rescued_vs_2000_low"], 1)
        self.assertEqual(result["harmed_vs_2000_low"], 1)
        self.assertEqual(result["rescue_fraction_global_eligible"], 1.)

    def test_macro_auc_weights_images_equally_not_action_pairs(self) -> None:
        # First image has one positive (8 pairs), second has four (20 pairs).
        rows = [fixture([1] + [0] * 8), fixture([1] * 4 + [0] * 5)]
        predictions = [[.9] + [.1] * 8, [.1] * 4 + [.9] * 5]
        result = scoring_metrics(rows, list(range(9)), predictions, False)
        self.assertEqual(result["within_image_macro_auc_mixed_only"], .5)
        self.assertEqual(result["mixed_candidate_images"], 2)

    def test_image_bootstrap_unit_count(self) -> None:
        interval = image_mean_interval([.25, .75], repeats=20)
        self.assertEqual(interval["n_images"], 2)
        self.assertEqual(interval["mean"], .5)
        self.assertIsNone(image_mean_interval([])["image_bootstrap_95_percentile_interval"])

    def test_original_answer_normalization(self) -> None:
        self.assertEqual(normalize("  2. "), "two")
        self.assertEqual(normalize(" BLUE "), "blue")
        self.assertNotEqual(normalize("a blue car"), normalize("blue"))


if __name__ == "__main__":
    unittest.main()
