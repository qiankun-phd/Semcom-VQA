"""EXP-014 trainer tests on synthetic features/outcomes, never experiment data."""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_selectors as training

try:
    import torch
except ImportError:
    torch = None


def protocol() -> dict:
    return json.loads(Path(__file__).with_name("protocol.json").read_text())


class ContractTests(unittest.TestCase):
    def test_protocol_exact_fit_and_recipe_budget(self) -> None:
        value = protocol()
        training.validate_protocol(value)
        self.assertEqual(len(training.RECIPES) * len(training.FAMILIES) * len(training.SEEDS), 84)
        self.assertEqual(len(training.RECIPES) * len(training.SEEDS), 12)
        for key, replacement in (("maximum_fits", 97), ("wallclock_limit_seconds", 3601), ("seeds", [7, 17])):
            changed = deepcopy(value)
            changed["training"][key] = replacement
            with self.assertRaises(ValueError):
                training.validate_protocol(changed)

    def test_family_reference_is_local_lowest_resource_action(self) -> None:
        self.assertEqual(training.reference_action(training.FAMILIES["rate_at_medium"]), 1)
        self.assertEqual(training.reference_action(training.FAMILIES["rate_at_high"]), 2)
        self.assertEqual(training.reference_action(training.FAMILIES["compute_at_4000"]), 3)
        self.assertEqual(training.reference_action([7, 1, 4]), 1)
        with self.assertRaises(ValueError):
            training.reference_action([0, 0, 3])

    def test_score_mapping_ties_and_signed_harm(self) -> None:
        self.assertEqual(training.choose_action([-.2, 0., -.1], [7, 1, 4], "gain"), 1)
        self.assertEqual(training.choose_action([0., .1, -.2], [3, 4, 5], "gain"), 4)
        self.assertEqual(training.choose_action([.5, .5, .5], [5, 3, 4], "absolute"), 3)
        with self.assertRaises(ValueError):
            training.choose_action([0., -.1, .1], [7, 1, 4], "gain")
        with self.assertRaises(ValueError):
            training.choose_action([-.1, .5, .7], [0, 3, 6], "absolute")

    def test_candidate_utility_then_cost_then_lexicographic_ties(self) -> None:
        rows = [
            {"recipe": "z", "family": "joint9", "summary": {"utility": .6, "mean_nominal_cost": 1.}},
            {"recipe": "a", "family": "joint9", "summary": {"utility": .5, "mean_nominal_cost": .5}},
            {"recipe": "c", "family": "joint9", "summary": {"utility": .6, "mean_nominal_cost": .8}},
            {"recipe": "b", "family": "joint9", "summary": {"utility": .6, "mean_nominal_cost": .8}},
        ]
        self.assertEqual(training.select_candidate(rows)["recipe"], "b")

    def test_nested_cannot_become_primary_and_axes_select_across_recipes(self) -> None:
        report = {"fixed": {str(a): {"utility": .5} for a in range(9)}, "evaluations": {"large": {}, "nested": {}}}
        for recipe in training.RECIPES:
            report["evaluations"]["large"][recipe] = {family: {"ensemble": {"utility": .5, "mean_nominal_cost": 1.}} for family in training.FAMILIES}
            report["evaluations"]["nested"][recipe] = {"joint9": {"ensemble": {"utility": 100., "mean_nominal_cost": .1}}}
        report["evaluations"]["large"]["original_gain"]["joint9"]["ensemble"]["utility"] = .6
        report["evaluations"]["large"]["balanced_gain"]["rate_at_high"]["ensemble"]["utility"] = .7
        result = training.validation_selection(report)
        self.assertEqual(result["primary_joint"]["arm"], "large")
        self.assertEqual(result["primary_joint"]["recipe"], "original_gain")
        self.assertEqual(result["strong_rate"]["recipe"], "balanced_gain")
        self.assertEqual(result["strong_rate"]["family"], "rate_at_high")
        self.assertEqual(result["strong_fixed_action"], 0)

    def test_wall_clock_interrupt_and_restore(self) -> None:
        with self.assertRaises(TimeoutError):
            with training.wall_clock_budget(.01):
                time.sleep(.1)
        with training.wall_clock_budget(.1):
            pass

    def test_prepare_allowlist_never_reads_or_hashes_test_data(self) -> None:
        manifest = {}
        for split, per_type, base in (("train", 800, 0), ("validation", 200, 10000)):
            manifest[split] = [{"id": f"{split}-{t}-{i}", "question_type": task, "image_id": base + t * per_type + i}
                               for t, task in enumerate(training.TASKS) for i in range(per_type)]
        manifest["nested_train"] = [row for row in manifest["train"] if int(row["id"].rsplit("-", 1)[1]) < 80]
        data = {f"{split}_manifest.json": rows for split, rows in manifest.items()}
        data.update({f"{split}_truth.json": [{"id": row["id"], "answer": "yes"} for row in rows] for split, rows in manifest.items()})
        data["features.json"] = [{"id": row["id"], "split": split} for split in ("train", "validation") for row in manifest[split]]
        data["features_complete.json"] = {"features_sha256": "same"}
        data["supervision_records.json"] = [{"id": row["id"], "split": split, "image_id": row["image_id"]} for split in ("train", "validation") for row in manifest[split] for _ in range(9)]
        data["supervision_complete.json"] = {"state": "SUPERVISION_COMPLETE", "records": 54000,
            "records_sha256": "same", "protocol_sha256": "same", "receiver_sha256": protocol()["receiver_sha256"]}
        data["frozen_data.json"] = {"state": "COMPLETE", "protocol_sha256": "same", "sha256": {},
            "truth_sha256": {f"{split}_truth.json": "same" for split in manifest}}
        data["protocol.json"] = protocol()
        accessed = []

        def safe_read(path: Path):
            self.assertFalse(path.name.startswith("test_"), path)
            accessed.append(path.name)
            return data[path.name]

        def safe_sha(path: Path):
            self.assertFalse(path.name.startswith("test_"), path)
            return "same"

        frozen = SimpleNamespace(validate_features=lambda rows, expected: {row["id"]: row for row in rows}, score_grid=lambda *args: {})
        with tempfile.TemporaryDirectory() as temporary, patch.object(training, "read", side_effect=safe_read), patch.object(training, "sha", side_effect=safe_sha):
            rows, _, _, _ = training.prepare_data(Path(temporary), Path(temporary) / "protocol.json", frozen)
        self.assertEqual(len(rows["nested_train"]), 480)
        self.assertEqual(len(rows["train"]), 4800)
        self.assertIn("nested_train_truth.json", accessed)


class GridDependencyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.grid = self.root / "grid_trainval"
        self.grid.mkdir()
        self.protocol_path = self.root / "protocol.json"
        training.save(self.protocol_path, protocol())
        self.receiver_hash = protocol()["receiver_sha256"]
        for name in ("frozen_data.json", "old_helper.py", "codec_weight.bin", "adapter_model.safetensors"):
            (self.root / name).write_text("synthetic input")
        common = {str(self.protocol_path): training.sha(self.protocol_path),
                  str(self.root / "old_helper.py"): training.sha(self.root / "old_helper.py")}
        self.codec = {"phase": "trainval", "controller_sha256": None, "sha256": {
            **common, str(self.root / "codec_weight.bin"): training.sha(self.root / "codec_weight.bin")}}
        self.inference = {"phase": "trainval", "controller_sha256": None, "sha256": {
            **common, str(self.root / "adapter_model.safetensors"): training.sha(self.root / "adapter_model.safetensors")}}
        training.save(self.grid / "codec_frozen.json", self.codec)
        training.save(self.grid / "inference_frozen.json", self.inference)
        training.save(self.grid / "representations.json", [])
        training.save(self.root / "supervision_records.json", [])
        training.save(self.grid / "model_runtime.json", {"gpu": "synthetic", "versions": {}})
        common_completion = {"phase": "trainval", "protocol_sha256": training.sha(self.protocol_path), "controller_sha256": None}
        self.encoding_complete = {**common_completion, "state": "COMPLETE", "representations": 18000,
            "representations_sha256": training.sha(self.grid / "representations.json")}
        self.inference_complete = {**common_completion, "state": "SUPERVISION_COMPLETE", "records": 54000,
            "records_sha256": training.sha(self.root / "supervision_records.json"), "receiver_sha256": self.receiver_hash,
            "labels_loaded": False, "old_test300_opened": False}
        training.save(self.grid / "encoding_complete.json", self.encoding_complete)
        training.save(self.root / "supervision_complete.json", self.inference_complete)

    def dependencies(self) -> dict:
        return training.training_grid_dependencies(self.root, self.protocol_path, self.receiver_hash)

    def test_pins_both_stage_freezes_all_transitive_dependencies_and_completed_artifacts(self) -> None:
        result = self.dependencies()
        for path in (self.grid / "codec_frozen.json", self.grid / "inference_frozen.json", self.root / "old_helper.py",
                     self.root / "codec_weight.bin", self.root / "adapter_model.safetensors",
                     self.grid / "representations.json", self.grid / "model_runtime.json", self.root / "supervision_complete.json"):
            self.assertEqual(result[str(path)], training.sha(path))

    def test_changed_historical_dependency_is_not_silently_re_frozen(self) -> None:
        (self.root / "old_helper.py").write_text("changed historical code")
        with self.assertRaisesRegex(ValueError, "Frozen dependency changed"):
            self.dependencies()

    def test_stage_phase_or_protocol_mismatch_rejected(self) -> None:
        for field, value in (("phase", "test"), ("controller_sha256", "unexpected")):
            changed = deepcopy(self.inference)
            changed[field] = value
            training.save(self.grid / "inference_frozen.json", changed)
            with self.assertRaises(ValueError):
                self.dependencies()
        changed = deepcopy(self.inference)
        changed["sha256"][str(self.protocol_path)] = "wrong"
        training.save(self.grid / "inference_frozen.json", changed)
        with self.assertRaisesRegex(ValueError, "current protocol"):
            self.dependencies()

    def test_stage_completion_must_have_matching_inventory_receiver_and_no_labels(self) -> None:
        for field, value in (("state", "SMOKE_COMPLETE"), ("records", 54), ("receiver_sha256", "wrong"),
                             ("labels_loaded", True), ("old_test300_opened", True)):
            changed = deepcopy(self.inference_complete)
            changed[field] = value
            training.save(self.root / "supervision_complete.json", changed)
            with self.assertRaises(ValueError):
                self.dependencies()
        training.save(self.root / "supervision_complete.json", self.inference_complete)
        changed = {**self.encoding_complete, "representations": 18}
        training.save(self.grid / "encoding_complete.json", changed)
        with self.assertRaises(ValueError):
            self.dependencies()

    def test_sealed_truth_dependency_rejected_before_file_access(self) -> None:
        forbidden = self.root / "test_truth.sealed.json"
        self.assertFalse(forbidden.exists())
        changed = deepcopy(self.inference)
        changed["sha256"][str(forbidden)] = "must not be read"
        training.save(self.grid / "inference_frozen.json", changed)
        with self.assertRaisesRegex(ValueError, "Test/answer"):
            self.dependencies()


@unittest.skipIf(torch is None, "Small tensor tests require torch")
class TensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(2)

    @staticmethod
    def factory() -> SimpleNamespace:
        def make_model(heads: int):
            return torch.nn.Sequential(torch.nn.Linear(339, 128), torch.nn.ReLU(), torch.nn.Dropout(.1),
                torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Dropout(.1), torch.nn.Linear(64, heads))
        return SimpleNamespace(make_model=make_model, vector=lambda row, scaler: row["vector"])

    def test_family_signed_targets_ignore_unavailable_global_zero(self) -> None:
        labels = torch.tensor([[1., 0., 0., 1., 1., 0., 0., 0., 0.]])
        values, column = training.gain_targets(labels, [7, 1, 4])
        self.assertEqual(column, 1)
        self.assertTrue(torch.equal(values, torch.tensor([[0., 0., 1.]])))
        labels[0, 1] = 1.
        values, _ = training.gain_targets(labels, [7, 1, 4])
        self.assertEqual(values[0, 0].item(), -1.)

    def test_reference_output_zero_and_receives_all_contrast_gradients(self) -> None:
        logits = torch.tensor([[.3, .5, -.4]], requires_grad=True)
        values = training.gain_outputs(logits, 1)
        self.assertEqual(values[0, 1].item(), 0.)
        torch.nn.functional.mse_loss(values[:, [0, 2]], torch.ones(1, 2)).backward()
        self.assertNotEqual(logits.grad[0, 1].item(), 0.)
        self.assertAlmostEqual(logits.grad.sum().item(), 0., places=6)

    def test_block_scaling_preserves_question_and_relative_image_norm(self) -> None:
        rows = [{"vector": [1 / 16] * 256 + [image_value] * 83} for image_value in (1., 2.)]
        original = training.feature_tensor(self.factory(), rows, {}, False)
        balanced = training.feature_tensor(self.factory(), rows, {}, True)
        self.assertTrue(torch.equal(original[:, :256], balanced[:, :256]))
        self.assertTrue(torch.allclose(original[:, 256:] / math.sqrt(83), balanced[:, 256:]))
        self.assertAlmostEqual(balanced[1, 256:].norm().item(), 2., places=6)

    def test_three_head_gain_fit_deterministic_and_best_checkpoint(self) -> None:
        generator = torch.Generator().manual_seed(99)
        x, y = torch.randn(12, 339, generator=generator), torch.randint(0, 2, (12, 9), generator=generator).float()
        settings = {"learning_rate": .001, "weight_decay": .0001, "batch_size": 4, "max_epochs": 3, "min_epochs": 2, "patience": 2}
        first, history, stats = training.fit_gain(self.factory(), x[:8], y[:8], x[8:], y[8:], [7, 1, 4], 7, settings)
        second, _, _ = training.fit_gain(self.factory(), x[:8], y[:8], x[8:], y[8:], [7, 1, 4], 7, settings)
        with torch.inference_mode():
            self.assertTrue(torch.equal(first(x), second(x)))
            target, column = training.gain_targets(y[8:], [7, 1, 4])
            observed = torch.nn.functional.mse_loss(training.gain_outputs(first(x[8:]), column)[:, [0, 2]], target[:, [0, 2]]).item()
        self.assertEqual(first(x).shape[1], 3)
        self.assertFalse(first.training)
        self.assertAlmostEqual(observed, stats["best_validation_gain_mse"], places=7)
        self.assertEqual(stats["best_validation_gain_mse"], min(row["validation_gain_mse"] for row in history))

    def test_minimum_epochs_even_when_validation_plateaus(self) -> None:
        settings = {"learning_rate": 0., "weight_decay": 0., "batch_size": 2, "max_epochs": 10, "min_epochs": 4, "patience": 2}
        x, y = torch.zeros(6, 339), torch.zeros(6, 9)
        _, history, stats = training.fit_gain(self.factory(), x[:4], y[:4], x[4:], y[4:], [3, 4, 5], 17, settings)
        self.assertEqual(len(history), 4)
        self.assertEqual(stats["best_epoch"], 1)


if __name__ == "__main__":
    unittest.main()
