"""Synthetic EXP-013 statistical and cache-contract tests; no dataset access."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

if __package__:
    from .evaluate_factorial import (
        CELLS, CONTRASTS, GROUPS, LABELS, evaluate_split, holm_adjust, paired_contrast,
        require_pinned_input, resolve_pinned_file, score_grid, select_action, sha, validate_predictions, verify_frozen, within_image_auc,
    )
else:
    from evaluate_factorial import (
        CELLS, CONTRASTS, GROUPS, LABELS, evaluate_split, holm_adjust, paired_contrast,
        require_pinned_input, resolve_pinned_file, score_grid, select_action, sha, validate_predictions, verify_frozen, within_image_auc,
    )


def fixture() -> tuple[dict, dict, dict, dict]:
    records = []
    patterns = {"a": {0, 1, 3, 4}, "b": {4, 8}, "c": set(range(9)), "d": set()}
    for identity, correct in patterns.items():
        for action, (budget, tier) in enumerate(CELLS):
            records.append({"id": identity, "image_id": list(patterns).index(identity), "question_type": "counting",
                            "budget": budget, "tier": tier, "prediction": "two" if action in correct else "three",
                            "codec_image_bytes": budget - 1, "image_bytes": budget,
                            "actual_visual_tokens": {"low": 50, "medium": 100, "high": 200}[tier],
                            "receiver_sha256": "receiver", "protocol_sha256": "protocol"})
    truth = [{"id": identity, "answer": "2"} for identity in patterns]
    grid = score_grid(records, truth, list(patterns), "receiver", "protocol")
    predictions = {}
    for identity in patterns:
        predictions[identity] = {"scores": {}}
        for group in GROUPS:
            scores = [.2] * 9 if group.endswith("absolute") else [0.] + [-.2] * 8
            predictions[identity]["scores"][group] = {seed: scores[:] for seed in LABELS}
    report = {"selection": {"strong_fixed_action": 3, "rate_variant": "rate_at_low", "compute_variant": "compute_at_4000"},
              "variants": {variant: {seed: {"selected": [rows[action] for rows in grid.values()]} for seed in LABELS}
                           for variant, action in (("rate_at_low", 0), ("compute_at_4000", 3))}}
    screen = {"maximum_lost_questions_vs_primary_fixed": 1, "maximum_mean_image_byte_ratio_vs_primary_fixed": .9,
              "minimum_utility_gain_over_strong_fixed_and_learned_single_axes": .005, "minimum_consistent_seeds": 2}
    return grid, predictions, report, screen


class EvaluationTest(unittest.TestCase):
    def test_signed_gain_is_valid_and_reference_zero_required(self) -> None:
        self.assertEqual(select_action([0.] + [-.5] * 8, "gain"), 0)
        self.assertEqual(select_action([0., .4] + [-.5] * 7, "gain"), 1)
        with self.assertRaises(ValueError):
            select_action([-.1] * 9, "gain")
        with self.assertRaises(ValueError):
            select_action([-.1] * 9, "absolute")

    def test_auc_ties_and_single_outcome(self) -> None:
        self.assertEqual(within_image_auc([1, 0, 1, 0], [1, 1, 0, 0]), .5)
        self.assertIsNone(within_image_auc([1, 2], [1, 1]))

    def test_float32_ensemble_rounding_is_allowed(self) -> None:
        _, predictions, _, _ = fixture()
        row = predictions["a"]
        row.update(id="a", split="validation")
        for group in GROUPS:
            row["scores"][group]["ensemble"][1] += 2e-8
        validate_predictions({"schema_version": 1, "rows": [row]}, [{"id": "a", "split": "validation"}])
        row["scores"][GROUPS[0]]["ensemble"][1] += .001
        with self.assertRaisesRegex(ValueError, "average all three"):
            validate_predictions({"schema_version": 1, "rows": [row]}, [{"id": "a", "split": "validation"}])

    def test_all_groups_and_seeds_share_identical_mixed_set(self) -> None:
        summary, units = evaluate_split(*fixture())
        self.assertEqual(summary["mixed_outcome_n"], 2)
        for group in GROUPS:
            for seed in LABELS:
                self.assertEqual(set(units[group][seed]["auc"]), {"a", "b"})
                self.assertEqual(summary["groups"][group][seed]["within_image_auc"]["n"], 2)
        self.assertEqual(summary["groups"][GROUPS[0]]["system_screen"]["full_cost_gate"], "PENDING")

    def test_paired_contrast_aligns_identity_not_input_order(self) -> None:
        values = {group: {"a": 1., "b": 0.} for group in GROUPS}
        values[GROUPS[1]] = {"b": 0., "a": 1.}
        result = paired_contrast(values, CONTRASTS["scale"], draws=100)
        self.assertEqual(result["mean_difference"], 0)
        self.assertEqual(result["sign_flip_p"], 1)
        values[GROUPS[2]].pop("b")
        with self.assertRaises(ValueError):
            paired_contrast(values, CONTRASTS["scale"])

    def test_holm_is_single_family(self) -> None:
        rows = [{"sign_flip_p": p} for p in [.001, .02, .04, .9]]
        holm_adjust(rows)
        self.assertEqual([row["holm_p"] for row in rows], [.004, .06, .08, .9])
        self.assertEqual(rows[0]["holm_family_size"], 4)

    def test_legacy_header_added_once(self) -> None:
        grid, _, _, _ = fixture()
        records = [{**row, "image_bytes": row["codec_image_bytes"]} for rows in grid.values() for row in rows]
        truth = [{"id": identity, "answer": "2"} for identity in grid]
        legacy = score_grid(records, truth, list(grid), "receiver", "protocol", legacy=True)
        self.assertEqual(legacy["a"][0]["image_bytes"], grid["a"][0]["image_bytes"])
        self.assertEqual(legacy["a"][0]["ldpc_complex_symbols"], 510 * 42)

    def test_freeze_required_before_labels_and_relocation_verified(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "Freeze all controllers"):
                verify_frozen(root, root / "protocol.json")
            path = root / "small.json"
            path.write_text(json.dumps({"value": 1}))
            self.assertEqual(resolve_pinned_file("/another/host/small.json", sha(path), (root,)), path)
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                resolve_pinned_file("/another/host/small.json", "bad", (root,))

    def test_source_protocol_must_match_training_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            path.write_text('{"screen":{"minimum_gain":0.005}}')
            frozen = {"/original/code/protocol.json": sha(path)}
            require_pinned_input(path, frozen)
            path.write_text('{"screen":{"minimum_gain":0}}')
            with self.assertRaisesRegex(ValueError, "frozen input"):
                require_pinned_input(path, frozen)


if __name__ == "__main__":
    unittest.main()
