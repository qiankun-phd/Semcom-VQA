"""Synthetic deterministic-allocation, provenance and stage-boundary checks."""
from __future__ import annotations

from collections import Counter
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

if __package__:
    from . import prepare_data as data
    from . import features
else:
    import prepare_data as data
    import features


def fixture() -> tuple[list[dict], dict[int, dict]]:
    annotations, questions = [], {}
    for index, task in enumerate(data.TASKS):
        for offset in range(8):
            identity = index * 100 + offset + 1
            qid = identity * 10
            annotations.append({"image_id": identity, "question_id": qid, "question_type": task,
                                "answers": [{"answer": " YES "}, {"answer": "yes"}]})
            questions[qid] = {"image_id": identity, "question": f"Question {qid}?"}
    return annotations, questions


class SelectionTest(unittest.TestCase):
    def test_exact_matching_avoids_scarce_type_theft(self) -> None:
        candidates = {"a": {1: {}, 2: {}}, "b": {2: {}, 3: {}}, "c": {1: {}}}
        selected = data.allocate_images(candidates, 1, "fixture")
        self.assertEqual(selected["c"], [1])
        self.assertEqual(len({identity for values in selected.values() for identity in values}), 3)
        self.assertEqual(Counter({task: len(ids) for task, ids in selected.items()}), Counter(a=1, b=1, c=1))

    def test_hall_capacity_detects_overlap_shortfall(self) -> None:
        candidates = {"a": {1: {}}, "b": {1: {}}}
        report = data.capacity_report(candidates, 1)
        self.assertFalse(report["jointly_feasible"])
        self.assertEqual(report["minimum_slack"]["slack"], -1)
        with self.assertRaisesRegex(ValueError, "no resampling"):
            data.allocate_images(candidates, 1, "fixture")

    def test_reordered_input_same_selection_balanced_disjoint_and_answer_free(self) -> None:
        annotations, questions = fixture()
        a = data.eligible_questions(annotations, questions, {1, 101}, "val2014")
        b = data.eligible_questions(list(reversed(annotations)), questions, {1, 101}, "val2014")
        first, truth = data.select_source(a, "val2014", {"validation": 2, "test": 3}, Path("/fixture"))
        second, _ = data.select_source(b, "val2014", {"validation": 2, "test": 3}, Path("/fixture"))
        self.assertEqual(first, second)
        rows = sum(first.values(), [])
        self.assertEqual(len(rows), len({row["image_id"] for row in rows}))
        self.assertFalse({1, 101} & {row["image_id"] for row in rows})
        self.assertTrue(all("answer" not in row for row in rows))
        self.assertEqual(Counter(row["question_type"] for row in first["test"]), Counter({task: 3 for task in data.TASKS}))
        self.assertEqual({row["answer"] for rows in truth.values() for row in rows}, {"yes"})

    def test_excluded_answers_not_accessed_and_ambiguity_excluded(self) -> None:
        annotations, questions = fixture()
        del annotations[0]["answers"]
        annotations[1]["answers"] = [{"answer": "yes"}, {"answer": "no"}]
        candidates = data.eligible_questions(annotations, questions, {1}, "val2014")
        self.assertNotIn(1, candidates[data.TASKS[0]])
        self.assertNotIn(2, candidates[data.TASKS[0]])

    def test_nested_is_same_source_subset_with_original_ids(self) -> None:
        annotations, questions = fixture()
        candidates = data.eligible_questions(annotations, questions, set(), "train2014")
        manifests, truths = data.select_source(candidates, "train2014", {"train": 5}, Path("/fixture"))
        rows, truth = data.nested_rows(manifests["train"], truths["train"], 2)
        self.assertEqual(len(rows), 12)
        self.assertEqual({row["id"] for row in rows}, {row["id"] for row in truth})
        self.assertTrue({row["id"] for row in rows} < {row["id"] for row in manifests["train"]})
        self.assertTrue(all(row["split"] == "train" and row["source_split"] == "train2014" for row in rows))
        self.assertEqual(Counter(row["question_type"] for row in rows), Counter({task: 2 for task in data.TASKS}))
        again, _ = data.nested_rows(list(reversed(manifests["train"])), truths["train"], 2)
        self.assertEqual(rows, again)


class BoundaryTest(unittest.TestCase):
    def test_test_download_and_feature_stop_before_manifest_without_freeze(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = Path(__file__).with_name("protocol.json")
            with patch.object(data, "download_phase") as download, patch.object(data, "load_legacy") as legacy:
                with self.assertRaisesRegex(ValueError, "Controller must be frozen"):
                    data.prepare(root, root, root, protocol, root, "test")
                download.assert_not_called()
                legacy.assert_not_called()
            with patch.object(features, "verify_data") as metadata:
                with self.assertRaisesRegex(ValueError, "Controller must be frozen"):
                    features.build(root, protocol, root, "test")
                metadata.assert_not_called()

    def test_label_free_checks_reject_truth_without_opening_it(self) -> None:
        with patch.object(data, "sha") as digest:
            with self.assertRaisesRegex(ValueError, "Truth file"):
                data.check_hashes({"/never/test_truth.sealed.json": "digest"}, label_free=True)
            digest.assert_not_called()

    def test_immutable_and_hash_resume_guards(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            data.save(path, {"value": 1}, immutable=True)
            expected = data.sha(path)
            data.save(path, {"value": 1}, immutable=True)
            with self.assertRaisesRegex(ValueError, "Immutable"):
                data.save(path, {"value": 2}, immutable=True)
            data.check_hashes({str(path): expected})
            data.save(path, {"value": 3})
            with self.assertRaisesRegex(ValueError, "changed"):
                data.check_hashes({str(path): expected})

    def test_exact_duplicate_fails_without_changing_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [{"id": "one", "image_id": 1, "source_split": "val2014", "file": str(root / "one.jpg")},
                    {"id": "two", "image_id": 2, "source_split": "val2014", "file": str(root / "two.jpg")}]
            protocol = {"resources": {"download_workers": 1, "download_attempts_per_image": 3}}
            original = copy.deepcopy(rows)
            with patch.object(data, "cached_images", return_value={}), patch.object(data, "fetch_image", return_value=("same", "fixture")):
                with self.assertRaisesRegex(ValueError, "must not be replaced"):
                    data.download_phase(root, root, rows, "test", protocol, set())
            self.assertEqual(rows, original)

    def test_test_content_collision_with_train_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [{"id": "one", "image_id": 1, "source_split": "val2014", "file": str(root / "one.jpg")}]
            protocol = {"resources": {"download_workers": 1, "download_attempts_per_image": 3}}
            with patch.object(data, "cached_images", return_value={}), patch.object(data, "fetch_image", return_value=("train-digest", "fixture")):
                with self.assertRaisesRegex(ValueError, "duplicate"):
                    data.download_phase(root, root, rows, "test", protocol, {"train-digest"})

    def test_complete_controller_required_including_evaluation_code(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = root / "protocol.json"
            data.save(protocol, {"fixture": True})
            frozen = {"state": "FROZEN_BEFORE_TEST", "completed_fits": 96, "test_labels_opened": False,
                      "test_features_opened": False, "protocol_sha256": data.sha(protocol)}
            data.save(root / "controller_frozen.json", frozen)
            with self.assertRaisesRegex(ValueError, "evaluation code"):
                data.verify_controller(root, protocol)


class FeatureTest(unittest.TestCase):
    def test_fixed_feature_dimensions_and_metadata_guard(self) -> None:
        row = {"question_features": [0.] * 256, "image_features": [0.] * 83,
               "question_feature_seconds": 0., "image_feature_seconds": 0.}
        features.validate_features(row)
        row["answer"] = "yes"
        with self.assertRaisesRegex(ValueError, "Forbidden"):
            features.validate_features(row)
        row.pop("answer")
        row["image_features"][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "Malformed"):
            features.validate_features(row)

    def test_feature_functions_match_frozen_exp012_without_labels(self) -> None:
        from PIL import Image
        source_code = Path(__file__).resolve().parents[1] / "rgb_joint_selector"
        legacy = data.load_legacy(source_code, "features")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.jpg"
            Image.new("RGB", (32, 24), (24, 100, 160)).save(path, "JPEG")
            first = legacy.extract("How many people?", path)
            second = legacy.extract("How many people?", path)
            self.assertEqual(first["question_features"], second["question_features"])
            self.assertEqual(first["image_features"], second["image_features"])
            features.validate_features(first)


if __name__ == "__main__":
    unittest.main()
