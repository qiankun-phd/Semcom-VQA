"""Synthetic EXP-014 evaluation tests. No experiment data or fitted model runs."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_test as evaluation  # noqa: E402

HERE = Path(__file__).resolve().parent


def _resolve_source() -> Path:
    candidates = [
        HERE.parent / "rgb_joint_selector",
        HERE.parent.parent / "rgb_joint_selector_20260922" / "code",
        HERE.parent.parent / "rgb_joint_selector" / "code",
    ]
    for candidate in candidates:
        if (candidate / "train_selector.py").is_file():
            return candidate
    return HERE.parent / "rgb_joint_selector"


SOURCE = _resolve_source()
TASKS = ["object_presence", "counting", "color", "positional_reasoning", "scene_recognition", "activity_recognition"]


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def source_module() -> object:
    sys.path.insert(0, str(SOURCE))
    spec = importlib.util.spec_from_file_location("exp014_fixture_source", SOURCE / "train_selector.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture() -> tuple[dict, list[dict], dict, dict, dict, list[dict], list[dict]]:
    protocol = json.loads((HERE / "protocol.json").read_text())
    protocol["data"]["splits"]["test"] = 6
    protocol["data"]["per_type"]["test"] = 1
    manifest = [{"id": str(i), "image_id": i, "split": "test", "source_split": "val2014", "question_type": task}
                for i, task in enumerate(TASKS)]
    truth = [{"id": str(i), "answer": "two"} for i in range(6)]
    patterns = [{0, 1, 4}, {3, 4, 6}, set(range(9)), set(), {1, 2, 4}, {0, 3, 5}]
    records = []
    for row, correct in zip(manifest, patterns):
        for action, (budget, tier) in enumerate(evaluation.CELLS):
            records.append({**row, "budget": budget, "tier": tier, "codec_image_bytes": budget - 3,
                            "image_bytes": budget - 2, "actual_visual_tokens": {"low": 50, "medium": 100, "high": 200}[tier],
                            "prediction": "2" if action in correct else "three", "receiver_sha256": "receiver",
                            "protocol_sha256": "protocol"})
    grid = source_module().score_grid(records, truth, [r["id"] for r in manifest], "receiver", "protocol")
    selection = {"primary_joint": {"arm": "large", "recipe": "original_absolute", "family": "joint9"},
                 "strong_rate": {"arm": "large", "recipe": "balanced_gain", "family": "rate_at_medium"},
                 "strong_compute": {"arm": "large", "recipe": "original_gain", "family": "compute_at_4000"},
                 "strong_fixed_action": 4, "primary_fixed_action": 4, "selection_split": "validation"}
    scores = {}
    for arm, recipe, family in evaluation.required_variants(selection):
        actions = evaluation.FAMILIES[family]
        chosen = [0, 3, 0, 0, 1, 0] if (arm, recipe, family) == ("large", "original_absolute", "joint9") else [actions[0]] * 6
        values = []
        for action in chosen:
            output = [.1] * len(actions) if recipe.endswith("absolute") else [-.1] * len(actions)
            if recipe.endswith("gain"):
                output[0] = 0.
            if action != actions[0] or recipe.endswith("absolute"):
                output[actions.index(action)] = .9
            values.append(output)
        rounded = np.asarray(values, dtype=np.float32).tolist()
        scores[(arm, recipe, family)] = {replicate: deepcopy(rounded) for replicate in evaluation.REPLICATES}
    return protocol, manifest, grid, scores, selection, records, truth


def frozen_fixture(root: Path) -> tuple[Path, dict, object, dict]:
    protocol, manifest, grid, scores, selection, records, truth = fixture()
    protocol_path = root / "protocol.json"
    write(protocol_path, protocol)
    write(root / "selection.json", selection)
    write(root / "validation_predictions.json", {"rows": []})
    write(root / "scalers.json", {"scalers": {}, "balanced_multiplier": 1 / math.sqrt(83)})
    for split in ("train", "validation"):
        write(root / f"{split}_manifest.json", [{"id": split, "image_id": 100 if split == "train" else 101}])
    write(root / "image_hashes.json", {"train": "train-content", "validation": "validation-content"})
    write(root / "historical_available_image_hashes.json", {"past": "historical-content"})
    write(root / "frozen_data.json", {"sha256": evaluation.snapshot([
        root / "image_hashes.json", root / "historical_available_image_hashes.json"])})
    checkpoints = {}
    checkpoint_files = []
    for arm in ("large", "nested"):
        checkpoints[arm] = {}
        for recipe in evaluation.RECIPES:
            checkpoints[arm][recipe] = {}
            for family, actions in evaluation.FAMILIES.items():
                if arm == "nested" and family != "joint9":
                    continue
                checkpoints[arm][recipe][family] = {}
                for seed in evaluation.SEEDS:
                    path = root / "checkpoints" / f"{arm}-{recipe}-{family}-{seed}.pt"
                    path.parent.mkdir(exist_ok=True)
                    path.write_bytes(b"fixture: never loaded by torch")
                    checkpoint_files.append(path)
                    checkpoints[arm][recipe][family][str(seed)] = {
                        "path": str(path), "sha256": evaluation.sha(path), "actions": actions,
                        "reference_action": min(actions, key=lambda a: (evaluation.nominal_cost(a), a))}
    artifacts = [root / name for name in ("selection.json", "scalers.json", "validation_predictions.json")]
    artifacts += [protocol_path, Path(evaluation.__file__).resolve(), *checkpoint_files]
    sources = [HERE / "train_selectors.py", SOURCE / "train_selector.py", SOURCE / "prepare_data.py",
               root / "train_manifest.json", root / "validation_manifest.json", root / "frozen_data.json"]
    write(root / "test_manifest.json", manifest)
    write(root / "test_truth.sealed.json", truth)
    controller = {"state": "FROZEN_BEFORE_TEST", "completed_fits": 96, "seeds": list(evaluation.SEEDS),
                  "test_labels_opened": False, "test_features_opened": False, "test_records_opened": False,
                  "sha256": evaluation.snapshot(artifacts), "source_inputs_sha256": evaluation.snapshot(sources),
                  "evaluation_code_sha256": evaluation.snapshot([Path(evaluation.__file__).resolve()]),
                  "protocol_sha256": evaluation.sha(protocol_path), "selection": selection, "checkpoints": checkpoints,
                  "test_manifest_sha256": evaluation.sha(root / "test_manifest.json"),
                  "sealed_test_truth_sha256": evaluation.sha(root / "test_truth.sealed.json"), "data_root": str(root)}
    write(root / "controller_frozen.json", controller)
    controller_hash = evaluation.sha(root / "controller_frozen.json")
    write(root / "training_complete.json", {"state": "COMPLETE", "completed_fits": 96, "controller_sha256": controller_hash})
    image_hashes = {row["id"]: "test-content-" + row["id"] for row in manifest}
    write(root / "test_image_hashes.json", image_hashes)
    write(root / "test_data_frozen.json", {"state": "COMPLETE", "test": 6,
        "protocol_sha256": evaluation.sha(protocol_path), "controller_sha256": controller_hash,
        "test_labels_opened": False, "sha256": evaluation.snapshot([
            root / "test_manifest.json", root / "test_image_hashes.json", root / "controller_frozen.json", root / "frozen_data.json"])})
    features = [{"id": row["id"], "split": "test", "source_image_sha256": image_hashes[row["id"]], "question_features": [1.] + [0.] * 255,
                 "image_features": [0.] * 83, "question_feature_seconds": 0., "image_feature_seconds": 0.} for row in manifest]
    write(root / "test_features.json", features)
    write(root / "test_features_frozen.json", {"phase": "test", "labels_loaded": False,
        "sha256": evaluation.snapshot([root / "test_manifest.json", root / "test_image_hashes.json", root / "test_data_frozen.json",
                                        protocol_path, root / "controller_frozen.json"])})
    write(root / "test_features_complete.json", {"state": "COMPLETE", "phase": "test", "records": 6,
        "features_sha256": evaluation.sha(root / "test_features.json"), "protocol_sha256": evaluation.sha(protocol_path),
        "controller_sha256": controller_hash, "labels_loaded": False,
        "frozen_sha256": evaluation.sha(root / "test_features_frozen.json")})
    for row in records:
        row["protocol_sha256"] = evaluation.sha(protocol_path)
        row["receiver_sha256"] = protocol["receiver_sha256"]
    write(root / "test_records.json", records)
    write(root / "test_inference_complete.json", {"state": "SUPERVISION_COMPLETE", "phase": "test", "records": 54,
        "records_sha256": evaluation.sha(root / "test_records.json"), "protocol_sha256": evaluation.sha(protocol_path),
        "receiver_sha256": protocol["receiver_sha256"], "controller_sha256": controller_hash,
        "labels_loaded": False, "old_test300_opened": False})
    # The production protocol validator is separately tested by the trainer.
    # Tiny integration fixtures use six images and never load fitted checkpoints.
    trainer = SimpleNamespace(validate_protocol=lambda value: None, load_source_module=lambda path: source_module())
    return protocol_path, scores, trainer, controller


class StatisticalTests(unittest.TestCase):
    def test_family_gain_anchor_is_not_global_zero(self) -> None:
        actions = [3, 4, 5]
        self.assertEqual(evaluation.select_action([0., -.1, -.2], actions, "gain"), 3)
        self.assertEqual(evaluation.select_action([0., .2, -.2], actions, "gain"), 4)
        with self.assertRaisesRegex(ValueError, "exactly zero"):
            evaluation.select_action([.1, 0., 0.], actions, "gain")

    def test_auc_uses_mixed_images_and_half_ties(self) -> None:
        self.assertEqual(evaluation.within_image_auc([.7, .7, .3, .2], [1, 0, 1, 0]), .625)
        self.assertIsNone(evaluation.within_image_auc([.7, .2], [1, 1]))

    def test_primary_family_three_holm_and_zero_sign_flip(self) -> None:
        values = {key: {"sign_flip_p": p} for key, p in zip(evaluation.PRIMARY_CONTROLS, [.01, .04, .8])}
        evaluation.holm_three(values)
        self.assertEqual([values[key]["holm_p"] for key in evaluation.PRIMARY_CONTROLS], [.03, .08, .8])
        self.assertEqual(evaluation.paired_sign_flip([0.] * 5, draws=37)["sign_flip_p"], 1.)
        with self.assertRaisesRegex(ValueError, "exactly the three"):
            evaluation.holm_three({})

    def test_margins_scale_with_question_count(self) -> None:
        protocol = fixture()[0]
        for n, lost in ((1200, 10), (2400, 20)):
            def method(correct: int, utility: float) -> dict:
                return {"n": n, "correct": correct, "utility": utility, "mean_framed_image_bytes": 100.}
            joint = {rep: method(n - lost, .8) for rep in evaluation.REPLICATES}
            controls = {name: {rep: method(n, .7) for rep in evaluation.REPLICATES}
                        for name in (*evaluation.PRIMARY_CONTROLS, "fixed_4000_medium")}
            for row in controls["fixed_4000_medium"].values():
                row["mean_framed_image_bytes"] = 200.
            result = evaluation.point_screen(joint, controls, protocol["screen"])
            self.assertTrue(result["point_screen_pass"])
            joint["ensemble"]["correct"] -= 1
            self.assertFalse(evaluation.point_screen(joint, controls, protocol["screen"])["checks"]["accuracy_margin"])

    def test_seed_gate_uses_the_same_seed_controls(self) -> None:
        protocol = fixture()[0]
        row = {"n": 2400, "correct": 2400, "utility": .8, "mean_framed_image_bytes": 100.}
        joint = {rep: dict(row) for rep in evaluation.REPLICATES}
        controls = {name: {rep: {**row, "utility": .7, "mean_framed_image_bytes": 200.} for rep in evaluation.REPLICATES}
                    for name in (*evaluation.PRIMARY_CONTROLS, "fixed_4000_medium")}
        controls["strong_rate"]["7"]["utility"] = .9
        controls["strong_compute"]["17"]["utility"] = .9
        result = evaluation.point_screen(joint, controls, protocol["screen"])
        self.assertEqual(result["consistent_seeds"], [27])
        self.assertFalse(result["checks"]["seed_consistency"])

    def test_report_keeps_frozen_primary_and_shared_auc_images(self) -> None:
        protocol, manifest, grid, scores, selection, _, _ = fixture()
        result = evaluation.evaluate_predictions(protocol, manifest, grid, scores, selection)
        self.assertEqual(result["selection"], selection)
        self.assertEqual(result["primary_joint"]["ensemble"]["correct"], 5)
        self.assertEqual(result["primary_joint"]["ensemble"]["rescue_vs_2000_low"], 2)
        self.assertEqual(result["primary_joint"]["ensemble"]["harm_vs_2000_low"], 0)
        self.assertEqual(result["mixed_image_n"], 4)
        for arm in ("large", "nested"):
            for recipe in evaluation.RECIPES:
                for seed in evaluation.REPLICATES:
                    self.assertEqual(result["secondary_joint_variants"][arm][recipe][seed]["within_image_auc"]["n_images"], 4)
        self.assertEqual(set(result["primary_utility_comparisons"]), set(evaluation.PRIMARY_CONTROLS))
        self.assertEqual(result["primary_utility_comparisons"]["strong_fixed"]["holm_family_size"], 3)
        expected = sum(r["utility"] for r in [grid[str(i)][a] for i, a in enumerate([0, 3, 0, 0, 1, 0])]) / 6
        self.assertAlmostEqual(result["primary_joint"]["ensemble"]["utility"], expected)


class BoundaryTests(unittest.TestCase):
    def test_incomplete_controller_never_reads_test_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write(root / "controller_frozen.json", {"state": "TRAINING", "completed_fits": 95})
            touched = []
            original = evaluation.read
            def watched(path: Path) -> object:
                touched.append(path.name)
                if path.name in evaluation.TEST_INPUT_NAMES:
                    self.fail("Test input opened before controller validation")
                return original(path)
            with patch.object(evaluation, "read", side_effect=watched):
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    evaluation.verify_controller(root, root / "protocol.json", SOURCE, HERE / "train_selectors.py")
            self.assertEqual(touched, ["controller_frozen.json"])

    def test_changed_checkpoint_blocks_before_test_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol, _, _, controller = frozen_fixture(root)
            first = controller["checkpoints"]["large"]["original_absolute"]["joint9"]["7"]
            Path(first["path"]).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "dependency changed"):
                evaluation.verify_controller(root, protocol, SOURCE, HERE / "train_selectors.py")

    def test_new_grid_uses_existing_header_and_rejects_old_inference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol_path, _, _, controller = frozen_fixture(root)
            protocol = evaluation.read(protocol_path)
            _, _, grid, _ = evaluation.load_test_inputs(root, protocol, protocol_path, controller, source_module())
            self.assertEqual(grid["0"][0]["image_bytes"], 1998)
            self.assertEqual(grid["0"][0]["ldpc_complex_symbols"], 510 * 42)
            complete = evaluation.read(root / "test_inference_complete.json")
            complete["controller_sha256"] = "another controller"
            write(root / "test_inference_complete.json", complete)
            with self.assertRaisesRegex(ValueError, "provenance"):
                evaluation.load_test_inputs(root, protocol, protocol_path, controller, source_module())

    def test_completed_call_returns_identical_report_without_forward(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol, scores, trainer, _ = frozen_fixture(root)
            with patch.object(evaluation, "load_module", return_value=trainer), patch.object(evaluation, "forward_frozen", return_value=scores) as forward:
                first = evaluation.run(root, protocol, SOURCE, HERE)
                second = evaluation.run(root, protocol, SOURCE, HERE)
                self.assertEqual(first, second)
                self.assertEqual(forward.call_count, 1)
                self.assertTrue((root / "test_evaluation_complete.json").is_file())
                truth = evaluation.read(root / "test_truth.sealed.json")
                truth[0]["answer"] = "changed"
                write(root / "test_truth.sealed.json", truth)
                with self.assertRaisesRegex(ValueError, "Test input changed"):
                    evaluation.run(root, protocol, SOURCE, HERE)
                self.assertEqual(forward.call_count, 1)

    def test_partial_attempt_does_not_retry_or_open_test_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol, _, trainer, _ = frozen_fixture(root)
            write(root / "test_evaluation_started.json", {"state": "STARTED"})
            with patch.object(evaluation, "load_module", return_value=trainer), patch.object(evaluation, "load_test_inputs") as loader:
                with self.assertRaises(FileExistsError):
                    evaluation.run(root, protocol, SOURCE, HERE)
                loader.assert_not_called()

    def test_missing_completion_receipt_cannot_recompute(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write(root / "test_report.json", {})
            with self.assertRaisesRegex(ValueError, "Incomplete test publication"):
                evaluation.cached_report(root, "controller")


if __name__ == "__main__":
    unittest.main()
