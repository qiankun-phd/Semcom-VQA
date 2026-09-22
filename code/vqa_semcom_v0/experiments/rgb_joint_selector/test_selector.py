"""Unit tests for deployable selection, chronology and tiny deterministic fits."""
from __future__ import annotations

import copy
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    from . import train_selector as selector
    from . import route_frame
except ImportError:
    import train_selector as selector
    import route_frame


def feature(identity: str, split: str, image_value: float = 0.) -> dict:
    return {"id": identity, "split": split, "question_features": [1.] + [0.] * 255,
            "image_features": [image_value] * 83, "question_feature_seconds": .001,
            "image_feature_seconds": .002}


def raw_grid(identity: str, legacy: bool = False) -> list[dict]:
    result = []
    for budget, tier in selector.CELLS:
        row = {"id": identity, "budget": budget, "tier": tier, "prediction": "yes",
               "codec_image_bytes": budget - 2, "image_bytes": budget - 1,
               "actual_visual_tokens": {"low": 49, "medium": 98, "high": 196}[tier],
               "receiver_seconds": .1, "preprocessing_seconds": .01, "encode_seconds": .2,
               "decode_seconds": .1, "receiver_sha256": "model", "protocol_sha256": "protocol"}
        if legacy:
            row["image_bytes"] -= 1
            del row["codec_image_bytes"]
        result.append(row)
    return result


class SelectorContracts(unittest.TestCase):
    def test_all_nine_actions_encode_only_the_tier_in_wire_header(self) -> None:
        payload = b"unchanged-self-describing-codec-packet"
        bundle = {"models": [], "actions": list(range(9)), "scaler": {}, "question_only": False}
        for action, (budget, tier) in enumerate(selector.CELLS):
            descriptor = selector.action_descriptor(action)
            probabilities = [0.] * 9
            probabilities[action] = 1.
            with patch.object(selector, "predict_rows", return_value=([probabilities], [.001])):
                predicted = selector.predict_deployment(bundle, [0.] * 256, [0.] * 83)
            frame = route_frame.pack(payload, tier)
            self.assertEqual(predicted["action"], action)
            self.assertEqual(predicted["budget"], budget)
            self.assertEqual(descriptor["route_byte"], frame[0])
            self.assertEqual(predicted["route_byte"], frame[0])
            self.assertEqual(predicted["route_byte"], route_frame.TIERS.index(tier))
            self.assertEqual(route_frame.unpack(bytes([predicted["route_byte"]]) + payload), (tier, payload))
            self.assertEqual(len(frame), len(payload) + 1)

    def test_registered_protocol(self) -> None:
        protocol = selector.read(Path(__file__).with_name("protocol.json"))
        selector.validate_protocol(protocol)
        protocol["training"]["seeds"] = [7]
        with self.assertRaisesRegex(ValueError, "seeds"):
            selector.validate_protocol(protocol)

    def test_probability_cost_selection_and_ties(self) -> None:
        self.assertEqual(selector.choose_action([.8] * 9, list(range(9))), 0)
        self.assertEqual(selector.choose_action([.1, .99, .1], [0, 3, 6]), 3)
        with self.assertRaises(ValueError):
            selector.choose_action([float("nan")], [0])
        with self.assertRaises(ValueError):
            selector.choose_action([1.1], [0])

    def test_feature_scaler_training_only_question_unchanged(self) -> None:
        rows = [feature("a", "train", 0.), feature("b", "train", 2.)]
        scaler = selector.fit_scaler(rows)
        result = selector.vector(rows[0], scaler)
        self.assertEqual(result[:256], rows[0]["question_features"])
        self.assertEqual(result[256:], [-1.] * 83)
        self.assertEqual(selector.vector(rows[0], scaler, True)[256:], [0.] * 83)
        with self.assertRaisesRegex(ValueError, "Only selector training"):
            selector.fit_scaler([feature("v", "validation", 200.)])

    def test_features_reject_bad_dimensions_duplicates_and_missing(self) -> None:
        row = feature("a", "train")
        selector.validate_features([row], {"train": {"a"}})
        for bad in ([row, row], [], [{**row, "question_features": [1.]}], [{**row, "image_feature_seconds": -1}]):
            with self.assertRaises(ValueError):
                selector.validate_features(bad, {"train": {"a"}})

    def test_normalization_matches_existing_grid(self) -> None:
        self.assertEqual(selector.normalize("  3.  "), "three")
        self.assertEqual(selector.normalize("RED CAR!"), "red car")
        self.assertEqual(selector.normalize("21"), "21")
        self.assertNotEqual(selector.normalize("a cat"), selector.normalize("cat"))

    def test_grid_raw_vs_framed_and_all_nine_actions(self) -> None:
        truth = [{"id": "a", "answer": "yes"}]
        modern = selector.score_grid(raw_grid("a"), truth, ["a"], "model", "protocol")
        legacy = selector.score_grid(raw_grid("a", True), truth, ["a"], "model", "protocol", True)
        self.assertEqual(modern, legacy)
        self.assertTrue(all(row["correct"] == 1 for row in modern["a"]))
        self.assertEqual(modern["a"][0]["ldpc_complex_symbols"], 510 * math.ceil(1999 / 48))
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            selector.score_grid(raw_grid("a")[:-1], truth, ["a"], "model", "protocol")
        bad = raw_grid("a")
        bad[0]["image_bytes"] += 1
        with self.assertRaisesRegex(ValueError, "byte accounting"):
            selector.score_grid(bad, truth, ["a"], "model", "protocol")

    def test_joint_feature_cost_and_question_only_cost_differ(self) -> None:
        rows = [feature("a", "validation")]
        grid = selector.score_grid(raw_grid("a"), [{"id": "a", "answer": "yes"}], ["a"], "model", "protocol")
        joint = selector.evaluate_policy(rows, grid, [[.8] * 9], list(range(9)), [.005])
        question = selector.evaluate_policy(rows, grid, [[.8] * 9], list(range(9)), [.005], question_only=True)
        self.assertAlmostEqual(joint["summary"]["decision_seconds"]["mean"], .008)
        self.assertAlmostEqual(question["summary"]["decision_seconds"]["mean"], .006)
        self.assertEqual(joint["summary"]["full_cost_gate"], "PENDING")

    def test_fixed_policy_selection_uses_realized_utility(self) -> None:
        rows = [feature("a", "validation")]
        grid = selector.score_grid(raw_grid("a"), [{"id": "a", "answer": "yes"}], ["a"], "model", "protocol")
        self.assertEqual(selector.choose_best_fixed(rows, grid), 0)
        grid["a"][8]["utility"] = 2.
        self.assertEqual(selector.choose_best_fixed(rows, grid), 8)

    def test_bootstrap_is_paired_deterministic_and_reports_effect(self) -> None:
        result = selector.bootstrap_difference([1., 1., 0.], [0., 1., 0.], repeats=40)
        self.assertEqual(result, selector.bootstrap_difference([1., 1., 0.], [0., 1., 0.], repeats=40))
        self.assertAlmostEqual(result["paired_mean_difference"], 1 / 3)
        with self.assertRaises(ValueError):
            selector.bootstrap_difference([1.], [1., 0.])

    def test_legacy_truth_requires_prior_valid_freeze_and_rejects_test(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            truth = root / "dev_truth.sealed.json"
            selector.save(truth, [{"id": "legacy", "answer": "yes"}])
            with self.assertRaisesRegex(ValueError, "must be frozen"):
                selector.read_legacy_truth(truth, root)
            policy = root / "policy.json"
            selector.save(policy, {"selected": 1})
            selector.save(root / "controller_frozen.json", {"state": "FROZEN_BEFORE_LEGACY_DEV_LABELS", "sha256": {str(policy): selector.sha(policy)}})
            self.assertEqual(selector.read_legacy_truth(truth, root)[0]["id"], "legacy")
            with self.assertRaisesRegex(ValueError, "Only explicitly named"):
                selector.read_legacy_truth(root / "test_truth.sealed.json", root)
            selector.save(policy, {"selected": 2})
            with self.assertRaisesRegex(ValueError, "changed or missing"):
                selector.read_legacy_truth(truth, root)

    def test_gate_requires_all_conditions_and_never_unlocks_test(self) -> None:
        screen = selector.read(Path(__file__).with_name("protocol.json"))["screen"]
        def evaluated(correct: int, utility: float, size: int) -> dict:
            return {"summary": {"correct": correct, "utility": utility, "image_bytes": {"mean": size}}}
        methods = {"joint": evaluated(94, .79, 3000), "primary_fixed": evaluated(94, .76, 4000),
                   "strong_fixed": evaluated(93, .77, 4000), "rate_only": evaluated(93, .78, 3500),
                   "compute_only": evaluated(93, .77, 4000)}
        per_seed = {str(s): methods for s in selector.SEEDS}
        result = selector.gate(methods, per_seed, screen)
        self.assertTrue(result["preliminary_pass"])
        self.assertEqual(result["full_cost_gate"], "PENDING")
        self.assertFalse(result["expand_to_test_or_snr"])
        methods["joint"]["summary"]["correct"] = 90
        self.assertFalse(selector.gate(methods, per_seed, screen)["preliminary_pass"])


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch not available locally")
class TinyTorchTraining(unittest.TestCase):
    def test_deterministic_tiny_multilabel_fit_and_deployment(self) -> None:
        import torch
        torch.set_num_threads(2)
        torch.manual_seed(91)
        x = torch.randn(12, 339)
        y = torch.randint(0, 2, (12, 9)).float()
        settings = {"learning_rate": .001, "weight_decay": .0001, "batch_size": 4,
                    "max_epochs": 3, "min_epochs": 1, "patience": 2}
        model, history, stats = selector.train_one(x[:8], y[:8], x[8:], y[8:], list(range(9)), 7, settings)
        repeated, repeated_history, _ = selector.train_one(x[:8], y[:8], x[8:], y[8:], list(range(9)), 7, settings)
        self.assertEqual(history, repeated_history)
        self.assertTrue(all(torch.equal(value, repeated.state_dict()[key]) for key, value in model.state_dict().items()))
        self.assertTrue(1 <= stats["best_epoch"] <= len(history))
        with torch.inference_mode():
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x[8:]), y[8:]).item()
        self.assertAlmostEqual(loss, stats["best_validation_bce"], places=6)
        scaler = {"mean": [0.] * 83, "std": [1.] * 83}
        bundle = {"models": [model], "actions": list(range(9)), "scaler": scaler, "question_only": False}
        result = selector.predict_deployment(bundle, [1.] + [0.] * 255, [0.] * 83)
        self.assertIn(result["action"], range(9))
        self.assertEqual(result["route_byte"], route_frame.TIERS.index(result["tier"]))
        self.assertGreaterEqual(result["router_seconds"], 0.)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model.pt"
            selector.torch_save_atomic(path, model.state_dict())
            restored = selector.load_checkpoint(path, list(range(9)))
            self.assertTrue(all(torch.equal(v, restored.state_dict()[k]) for k, v in model.state_dict().items()))


if __name__ == "__main__":
    unittest.main()
