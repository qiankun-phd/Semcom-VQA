"""Lightweight provenance, answer-isolation and resumability tests."""
from __future__ import annotations

from collections import Counter
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    from . import prepare_data as data
    from . import encode_data as codec
except ImportError:
    import prepare_data as data
    import encode_data as codec


class DataSelectionTests(unittest.TestCase):
    def fixture(self) -> tuple[list[dict], dict[int, dict]]:
        annotations, questions = [], {}
        for index, task in enumerate(data.TASKS):
            for offset in range(10):
                iid = qid = index * 100 + offset + 1
                annotations.append({"image_id": iid, "question_id": qid, "question_type": task,
                                    "answers": [{"answer": " YES "}, {"answer": "yes"}]})
                questions[qid] = {"image_id": iid, "question": f"Question {qid}?"}
        return annotations, questions

    def test_deterministic_balanced_image_disjoint_and_answer_free_manifests(self) -> None:
        annotations, questions = self.fixture()
        rows, truths = data.select_rows(annotations, questions, {1, 101}, Path("/data"), {"train": 3, "validation": 2})
        repeated, _ = data.select_rows(list(reversed(annotations)), questions, {1, 101}, Path("/data"), {"train": 3, "validation": 2})
        self.assertEqual(rows, repeated)
        selected = [row for split in rows.values() for row in split]
        self.assertEqual(len(selected), len({r["image_id"] for r in selected}))
        self.assertFalse({r["image_id"] for r in selected} & {1, 101})
        self.assertTrue(all("answer" not in row for row in selected))
        self.assertEqual(Counter(r["question_type"] for r in rows["train"]), Counter({t: 3 for t in data.TASKS}))
        self.assertEqual({r["answer"] for truth in truths.values() for r in truth}, {"yes"})

    def test_excluded_annotation_answers_never_accessed(self) -> None:
        annotations, questions = self.fixture()
        del annotations[0]["answers"]  # Access would raise, including through normalizer.
        data.select_rows(annotations, questions, {1}, Path("/data"), {"train": 2, "validation": 1})

    def test_ambiguous_or_empty_answers_are_not_silently_normalized_to_one(self) -> None:
        self.assertIsNone(data.normalized_single_answer({"answers": [{"answer": "yes"}, {"answer": "no"}]}))
        self.assertIsNone(data.normalized_single_answer({"answers": [{"answer": " "}]}))
        self.assertIsNone(data.normalized_single_answer({"answers": []}))

    def test_insufficient_data_fails_not_smaller_or_resampled(self) -> None:
        annotations, questions = self.fixture()
        with self.assertRaisesRegex(ValueError, "Insufficient fresh"):
            data.select_rows(annotations, questions, set(), Path("/data"), {"train": 11})

    def test_identity_mismatch_fails(self) -> None:
        annotations, questions = self.fixture()
        for question in questions.values():
            question["image_id"] += 9000
        with self.assertRaisesRegex(ValueError, "identity disagreement"):
            data.select_rows(annotations, questions, set(), Path("/data"), {"train": 1})

    def test_inventory_never_includes_sealed_truth_and_excludes_own_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp)
            stage1, output = project / "outputs/stage1", project / "outputs/new"
            required = [project / "outputs/tdiuc_qlora_pilot_20260916/train.json"] + [
                stage1 / name for name in ("train_manifest.json", "dev_manifest.json", "test_manifest.json", "exclusion_audit.json")]
            for path in required:
                data.save(path, [])
            data.save(stage1 / "test_truth.sealed.json", {"never_read": True})
            data.save(stage1 / "label_manifest.json", [])
            data.save(output / "train_manifest.json", [])
            inventory = data.exclusion_sources(project, stage1, output)
            self.assertEqual(set(inventory), set(required))

    def test_frozen_hash_detects_changed_or_missing_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "source.json"
            data.save(path, {"a": 1})
            hashes = {str(path): data.sha(path)}
            data.check_hashes(hashes)
            data.save(path, {"a": 2})
            with self.assertRaisesRegex(ValueError, "changed or missing"):
                data.check_hashes(hashes)
            path.unlink()
            with self.assertRaises(ValueError):
                data.check_hashes(hashes)

    def test_frozen_cached_download_checks_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "COCO_val2014_000000000001.jpg"
            codec.publish_bytes(path, b"fake image")
            row = {"file": str(path), "image_id": 1,
                   "url": "https://s3.amazonaws.com/images.cocodataset.org/val2014/COCO_val2014_000000000001.jpg"}
            with patch.object(data, "validate_image"):
                self.assertEqual(data.fetch_image(row, data.sha(path)), data.sha(path))
                with self.assertRaisesRegex(ValueError, "Cached source image changed"):
                    data.fetch_image(row, "wrong")


class CodecJournalTests(unittest.TestCase):
    def record(self, root: Path) -> dict:
        packet, png = root / "a.bin", root / "a.png"
        codec.publish_bytes(packet, b"packet")
        codec.publish_bytes(png, b"png")
        return {"id": "one", "budget": 2000, "roundtrip": True, "codec_image_bytes": 6,
                "image_bytes": 7, "packet": str(packet), "decoded": str(png),
                "packet_sha256": data.sha(packet), "decoded_sha256": data.sha(png),
                "encode_seconds": .1, "decode_seconds": .2}

    def test_one_byte_route_cost_and_raw_payload_are_separate(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            record = self.record(Path(temp))
            codec.validate_record(record)
            for bad in (6, 8):
                record["image_bytes"] = bad
                with self.assertRaisesRegex(ValueError, "route overhead"):
                    codec.validate_record(record)

    def test_atomic_orphan_replay_does_not_overwrite_different_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "data.bin"
            codec.publish_bytes(path, b"first")
            codec.publish_bytes(path, b"first")
            with self.assertRaisesRegex(ValueError, "artifact differs"):
                codec.publish_bytes(path, b"second")
            self.assertEqual(path.read_bytes(), b"first")

    def test_orphan_journal_recovered_and_changed_record_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            record = self.record(root)
            data.save(root / "encoding_records/one-2000.json", record)
            self.assertEqual(codec.recover_records(root, [{"id": "one"}]), [record])
            changed = copy.deepcopy(record)
            changed["encode_seconds"] = .9
            data.save(root / "representations.json", [changed])
            with self.assertRaisesRegex(ValueError, "journals disagree"):
                codec.recover_records(root, [{"id": "one"}])

    def test_corrupted_output_or_out_of_grid_records_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            record = self.record(root)
            data.save(root / "representations.json", [record])
            with self.assertRaisesRegex(ValueError, "outside frozen"):
                codec.recover_records(root, [{"id": "two"}])
            data.save(Path(record["decoded"]), {"different": True})
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                codec.recover_records(root, [{"id": "one"}])

    def test_smoke_rows_balanced_deterministic(self) -> None:
        annotations, _ = DataSelectionTests().fixture()
        rows = [{**r, "id": f"item-{r['question_id']}"} for r in annotations]
        selected = codec.smoke_rows(rows)
        self.assertEqual(selected, codec.smoke_rows(list(reversed(rows))))
        self.assertEqual([r["question_type"] for r in selected], sorted(data.TASKS))


if __name__ == "__main__":
    unittest.main()
