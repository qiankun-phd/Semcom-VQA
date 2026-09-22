"""Small-tensor and boundary tests; these never touch experimental outcome data."""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_factorial as training

try:
    import torch
except ImportError:
    torch = None


class ContractTests(unittest.TestCase):
    def test_current_protocol_and_fixed_factor_design(self) -> None:
        protocol = json.loads(Path(__file__).with_name("protocol.json").read_text())
        training.validate_protocol(protocol)
        for key, value in (("maximum_fits", 13), ("seeds", [7, 17]), ("wall_clock_limit_seconds", 1801)):
            changed = deepcopy(protocol)
            changed["training"][key] = value
            with self.assertRaises(ValueError):
                training.validate_protocol(changed)
        changed = deepcopy(protocol)
        changed["features"]["balanced_image_multiplier"] = "tuned"
        with self.assertRaises(ValueError):
            training.validate_protocol(changed)

    def test_signed_gain_selection_preserves_harm_and_reference(self) -> None:
        self.assertEqual(training.choose_action([0] + [-.1] * 8, "gain"), 0)
        self.assertEqual(training.choose_action([0, -.2, -.2, .05, -.2, -.2, -.2, -.2, -.2], "gain"), 3)
        with self.assertRaises(ValueError):
            training.choose_action([.01] * 9, "gain")
        with self.assertRaises(ValueError):
            training.choose_action([0] + [-1.1] * 8, "gain")
        self.assertEqual(training.choose_action([.5] * 9, "absolute"), 0)

    def test_reproduction_all_seeds_ensemble_and_splits(self) -> None:
        rows, old_rows = [], []
        for split in training.EXPECTED_SPLITS:
            values = {replicate: [.5] * 9 for replicate in ("7", "17", "27", "ensemble")}
            rows.append({"id": split, "split": split, "scores": {"original_absolute": deepcopy(values)}})
            old_rows.append({"id": split, "split": split, "probabilities": {"joint9": deepcopy(values)}})
        baseline = {"actions": {"joint9": list(range(9))}, "rows": old_rows}
        result = training.reproduction_check(rows, baseline)
        self.assertTrue(result["passed"])
        self.assertEqual(result["checked_scores"], 3 * 4 * 9)
        self.assertEqual(result["checked_actions"], 3 * 4)
        rows[2]["scores"]["original_absolute"]["27"][8] += 2e-6
        self.assertFalse(training.reproduction_check(rows, baseline)["passed"])
        rows[2]["scores"]["original_absolute"]["27"][8] = .9
        result = training.reproduction_check(rows, baseline)
        self.assertEqual(result["action_mismatches"], 1)

    def test_reproduction_rejects_mixed_ids_or_mapping(self) -> None:
        baseline = {"actions": {"joint9": list(range(9))}, "rows": [{"id": "a"}, {"id": "a"}]}
        with self.assertRaises(ValueError):
            training.reproduction_check([], baseline)

    def test_wall_clock_interrupt_and_restore(self) -> None:
        with self.assertRaises(TimeoutError):
            with training.wall_clock_budget(.01):
                time.sleep(.1)
        with training.wall_clock_budget(.1):
            pass
        with self.assertRaises(TimeoutError):
            with training.wall_clock_budget(0):
                pass


@unittest.skipIf(torch is None, "Small-tensor tests require torch")
class TensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(2)

    @staticmethod
    def factory() -> SimpleNamespace:
        def make_model(heads: int):
            return torch.nn.Sequential(torch.nn.Linear(339, 128), torch.nn.ReLU(), torch.nn.Dropout(.1),
                                       torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Dropout(.1),
                                       torch.nn.Linear(64, heads))
        return SimpleNamespace(make_model=make_model, vector=lambda row, scaler: row["vector"])

    def test_block_scaling_keeps_question_and_does_not_normalize_examples(self) -> None:
        rows = [{"vector": [1 / 16] * 256 + [value] * 83} for value in (1., 2.)]
        original = training.feature_tensor(self.factory(), rows, {}, False)
        balanced = training.feature_tensor(self.factory(), rows, {}, True)
        self.assertTrue(torch.equal(original[:, :256], balanced[:, :256]))
        self.assertTrue(torch.allclose(original[:, 256:] / math.sqrt(83), balanced[:, 256:]))
        self.assertAlmostEqual(balanced[0, 256:].norm().item(), 1., places=6)
        self.assertAlmostEqual(balanced[1, 256:].norm().item(), 2., places=6)

    def test_gain_has_negative_targets_and_exact_zero_reference(self) -> None:
        correctness = torch.tensor([[1., 0., 1., 0., 0., 1., 0., 1., 0.], [0., 1., 0., 0., 0., 0., 1., 0., 0.]])
        gains = training.gain_targets(correctness)
        self.assertEqual(gains[0, 1].item(), -1.)
        self.assertEqual(gains[1, 1].item(), 1.)
        self.assertTrue(torch.equal(gains[:, 0], torch.zeros(2)))
        logits = torch.tensor([[.3, -.5, .7, .1, -.2, 1., -1., .5, .9]], requires_grad=True)
        outputs = training.gain_outputs(logits)
        self.assertEqual(outputs[0, 0].item(), 0.)
        self.assertTrue(torch.all(outputs.abs() <= 1))
        loss = torch.nn.functional.mse_loss(outputs[:, 1:], torch.ones(1, 8))
        loss.backward()
        self.assertNotEqual(logits.grad[0, 0].item(), 0.)
        self.assertAlmostEqual(logits.grad.sum().item(), 0., places=6)

    def test_small_gain_fit_is_deterministic_and_returns_best_validation_checkpoint(self) -> None:
        generator = torch.Generator().manual_seed(99)
        inputs = torch.randn(12, 339, generator=generator)
        labels = torch.randint(0, 2, (12, 9), generator=generator).float()
        settings = {"learning_rate": .001, "weight_decay": .0001, "batch_size": 4,
                    "max_epochs": 3, "min_epochs": 2, "patience": 2}
        first, history, stats = training.fit_gain(self.factory(), inputs[:8], labels[:8], inputs[8:], labels[8:], 7, settings)
        second, _, _ = training.fit_gain(self.factory(), inputs[:8], labels[:8], inputs[8:], labels[8:], 7, settings)
        self.assertFalse(first.training)
        self.assertEqual(stats["best_validation_gain_mse"], min(row["validation_gain_mse"] for row in history))
        with torch.inference_mode():
            self.assertTrue(torch.equal(first(inputs), second(inputs)))
            observed = torch.nn.functional.mse_loss(training.gain_outputs(first(inputs[8:]))[:, 1:], training.gain_targets(labels[8:])[:, 1:]).item()
        self.assertAlmostEqual(observed, stats["best_validation_gain_mse"], places=7)

    def test_early_stopping_honors_minimum_epochs(self) -> None:
        inputs, labels = torch.zeros(6, 339), torch.zeros(6, 9)
        settings = {"learning_rate": 0., "weight_decay": 0., "batch_size": 2,
                    "max_epochs": 10, "min_epochs": 4, "patience": 2}
        _, history, stats = training.fit_gain(self.factory(), inputs[:4], labels[:4], inputs[4:], labels[4:], 17, settings)
        self.assertEqual(len(history), 4)
        self.assertEqual(stats["best_epoch"], 1)


if __name__ == "__main__":
    unittest.main()
