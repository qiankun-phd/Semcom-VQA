"""Small synthetic tests for answer-assisted cache diagnosis, not dataset tests."""
from __future__ import annotations

import copy
import unittest

if __package__:
    from .opportunities import (
        CELLS, analyze_split, check_deployed, opportunity_flags, score_grid,
        summarize, summarize_oracles,
    )
else:
    from opportunities import (
        CELLS, analyze_split, check_deployed, opportunity_flags, score_grid,
        summarize, summarize_oracles,
    )


def cells(correct_actions: list[int], identity: str = "example") -> list[dict]:
    result = []
    for action, (budget, tier) in enumerate(CELLS):
        tokens = {"low": 50, "medium": 100, "high": 200}[tier]
        correct = int(action in correct_actions)
        result.append({"id": identity, "image_id": 123, "question_type": "counting",
                       "budget": budget, "tier": tier, "action": action,
                       "image_bytes": budget, "actual_visual_tokens": tokens,
                       "correct": correct, "prediction": "two" if correct else "three",
                       "utility": correct - .05 * (budget / 8001 + tokens / 200)})
    return result


class OpportunityTest(unittest.TestCase):
    def test_rescue_sets_include_overlaps_and_joint_required(self) -> None:
        overlap = opportunity_flags(cells([1, 3, 4]))
        self.assertTrue(overlap["all_three_intersection"])
        self.assertTrue(overlap["rate_or_compute_union"])
        self.assertFalse(overlap["both_axes_required"])
        joint_required = opportunity_flags(cells([4, 8]))
        self.assertTrue(joint_required["both_axes_required"])
        self.assertFalse(joint_required["rate_or_compute_union"])
        self.assertTrue(opportunity_flags(cells([]))["unrescuable_any_grid"])
        self.assertFalse(opportunity_flags(cells([0, 4]))["any_grid_rescuable"])

    def test_strong_axis_oracle_searches_all_fixed_values(self) -> None:
        grid = {"a": cells([7], "a"), "b": cells([4], "b")}
        summary, choices = summarize_oracles(grid, None)
        self.assertEqual(summary["strongest_fixed_other_axis"]["rate_only"]["selected_retrospectively_by_mean_utility"], "rate_at_medium")
        self.assertEqual(summary["strongest_fixed_other_axis"]["rate_only"]["summary"]["correct"], 2)
        self.assertEqual(summary["strongest_fixed_other_axis"]["compute_only"]["selected_retrospectively_by_mean_utility"], "compute_at_4000")
        self.assertEqual(choices["nine_way"], {"a": 7, "b": 4})
        self.assertEqual(summary["all_families"]["rate_at_low"]["correct"], 0)

    def test_nonmonotonicity_and_deployed_miss_decompose(self) -> None:
        grid = {"a": cells([0], "a"), "b": cells([4], "b"), "c": cells([], "c")}
        selected = [grid["a"][3], grid["b"][0], grid["c"][0]]
        report = {"selection": {"rate_variant": "rate_at_low", "compute_variant": "compute_at_2000", "strong_fixed_action": 0},
                  "methods": {"joint": {"selected": selected, "summary": summarize(selected)},
                              "rate_only": {"selected": selected, "summary": summarize(selected)}}}
        summary, private = analyze_split(grid, report)
        self.assertEqual(summary["grid_agreement"]["all_wrong"], 1)
        self.assertEqual(summary["grid_agreement"]["disagreement"], 2)
        self.assertEqual(summary["nonmonotonicity"]["baseline_correct_images_lost_by_any_increase"], 1)
        missed = summary["deployed_methods"]["joint"]["misses"]
        self.assertEqual(missed["missed_rescuable_answer"], 2)
        self.assertEqual(missed["baseline_correct_lost"], 1)
        self.assertEqual(missed["baseline_wrong_rescuable_missed"], 1)
        self.assertEqual(summary["deployed_methods"]["rate_only"]["misses"]["missed_only_outside_deployed_action_family"], 1)
        regret = summary["deployed_methods"]["joint"]["nine_way_oracle_regret"]["mean_utility_regret"]
        self.assertAlmostEqual(regret, missed["mean_correctness_regret"] + missed["mean_resource_cost_regret_component"])
        self.assertEqual(len(private), 3)
        self.assertNotIn('"id"', __import__("json").dumps(summary))

    def test_original_scoring_handles_legacy_header_exactly_once(self) -> None:
        records = []
        for action, (budget, tier) in enumerate(CELLS):
            records.append({"id": "sample", "question_type": "counting", "budget": budget, "tier": tier,
                            "image_bytes": budget - 1, "actual_visual_tokens": {"low": 50, "medium": 100, "high": 200}[tier],
                            "prediction": "2", "receiver_sha256": "receiver", "protocol_sha256": "protocol"})
        legacy = score_grid(records, [{"id": "sample", "answer": "two"}], ["sample"], "receiver", "protocol", legacy=True)
        self.assertEqual(legacy["sample"][0]["image_bytes"], 2000)
        self.assertEqual(legacy["sample"][0]["codec_image_bytes"], 1999)
        self.assertEqual(legacy["sample"][0]["correct"], 1)
        modern = copy.deepcopy(records)
        for row in modern:
            row["codec_image_bytes"] = row["image_bytes"]
            row["image_bytes"] += 1
        scored = score_grid(modern, [{"id": "sample", "answer": "two"}], ["sample"], "receiver", "protocol")
        self.assertEqual(legacy, scored)

    def test_rejects_altered_deployed_scores(self) -> None:
        grid = {"a": cells([0], "a")}
        selected = [copy.deepcopy(grid["a"][0])]
        selected[0]["correct"] = 0
        with self.assertRaisesRegex(ValueError, "cached score mismatch"):
            check_deployed({"methods": {"joint": {"selected": selected}}}, grid)


if __name__ == "__main__":
    unittest.main()
