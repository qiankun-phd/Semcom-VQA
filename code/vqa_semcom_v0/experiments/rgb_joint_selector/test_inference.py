"""Fast inference integrity tests. No model, dataset, answers, or GPU required."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from infer_supervision import ordered_rows, verify_cached_records, verify_frozen_data, verify_packets


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixtures() -> dict[str, list[dict]]:
    return {split: [{"id": f"joint-selector-{split}-{index}", "image_id": offset + index,
                    "question_type": str(index % 6), "question": "Is this visible?"}
                   for index in range(12)] for split, offset in (("train", 0), ("validation", 100))}


class InferenceTests(unittest.TestCase):
    def test_balanced_label_free_order(self) -> None:
        rows = ordered_rows(fixtures(), 42)
        self.assertEqual(len(rows), 24)
        self.assertEqual({row["split"] for row in rows[:6]}, {"train"})
        self.assertEqual(len({row["question_type"] for row in rows[:6]}), 6)
        reordered = {split: list(reversed(rows)) for split, rows in fixtures().items()}
        self.assertEqual(rows, ordered_rows(reordered, 42))

    def test_no_labels_or_test_or_duplicate_images(self) -> None:
        for mutation in ({"answer": "yes"}, {"id": "joint-test-1"}, {"split": "test"}, {"image_id": 100}):
            manifests = fixtures()
            manifests["train"][0].update(mutation)
            with self.assertRaises(ValueError):
                ordered_rows(manifests, 42)
        with self.assertRaises(ValueError):
            ordered_rows({**fixtures(), "test": []}, 42)

    def test_packet_framing_and_growth(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            packet, decoded = root / "packet.bin", root / "decoded.png"
            packet.write_bytes(b"codec")
            decoded.write_bytes(b"decoded-fixture")
            record = {"id": "a", "budget": 2000, "codec_image_bytes": 5, "image_bytes": 6,
                      "roundtrip": True, "packet": str(packet), "decoded": str(decoded),
                      "packet_sha256": sha(packet), "decoded_sha256": sha(decoded)}
            selected = verify_packets([record, {**record, "id": "later"}], [{"id": "a"}], [2000], sha)
            self.assertEqual(set(selected), {("a", 2000)})
            for bad in ({**record, "image_bytes": 5}, {**record, "codec_image_bytes": 6},
                        {**record, "roundtrip": False}, {**record, "decoded_sha256": "wrong"}):
                with self.assertRaises(ValueError):
                    verify_packets([bad], [{"id": "a"}], [2000], sha)
            with self.assertRaises(ValueError):
                verify_packets([record, record], [{"id": "a"}], [2000], sha)
            packet.write_bytes(b"mutated")
            with self.assertRaises(ValueError):
                verify_packets([record], [{"id": "a"}], [2000], sha)

    def test_frozen_data_does_not_open_truth(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            paths = {split: root / f"{split}_manifest.json" for split in ("train", "validation")}
            for path in [*paths.values(), root / "image_hashes.json", root / "protocol.json"]:
                path.write_text(json.dumps({"fixture": path.name}), encoding="utf-8")
            frozen = {"state": "COMPLETE", "train": 480, "validation": 120,
                      "protocol_sha256": sha(root / "protocol.json"), "sha256": {
                          str(path): sha(path) for path in [*paths.values(), root / "image_hashes.json"]}}
            frozen["sha256"][str(root / "nonexistent_truth.sealed.json")] = "must-not-open"
            verify_frozen_data(frozen, root / "protocol.json", paths, [], sha)
            paths["train"].write_text("changed", encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_frozen_data(frozen, root / "protocol.json", paths, [], sha)

    def test_exact_codec_cap_plus_header_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            decoded = root / "decoded.png"
            decoded.write_bytes(b"decoded-fixture")
            for budget in (2000, 4000, 8000):
                packet = root / f"{budget}.bin"
                packet.write_bytes(b"x" * budget)
                representation = {"id": "a", "budget": budget, "codec_image_bytes": budget,
                                  "image_bytes": budget + 1, "roundtrip": True,
                                  "packet": str(packet), "decoded": str(decoded),
                                  "packet_sha256": sha(packet), "decoded_sha256": sha(decoded)}
                selected = verify_packets([representation], [{"id": "a"}], [budget], sha)
                record = {**representation, "tier": "low", "receiver_sha256": "r", "protocol_sha256": "p",
                          "prediction": "yes", "actual_visual_tokens": 48,
                          "receiver_seconds": 0.1, "preprocessing_seconds": 0.01}
                self.assertEqual(verify_cached_records([record], {("a", budget, "low")}, selected, "r", "p"),
                                 {("a", budget, "low")})
                packet.write_bytes(b"x" * (budget + 1))
                over_cap = {**representation, "codec_image_bytes": budget + 1, "image_bytes": budget + 2,
                            "packet_sha256": sha(packet)}
                with self.assertRaises(ValueError):
                    verify_packets([over_cap], [{"id": "a"}], [budget], sha)

    def test_resume_rejects_representation_mutation(self) -> None:
        representation = {"packet_sha256": "p", "decoded_sha256": "d", "image_bytes": 6, "codec_image_bytes": 5}
        record = {"id": "a", "budget": 2000, "tier": "low", "receiver_sha256": "r", "protocol_sha256": "x",
                  "prediction": "yes", "actual_visual_tokens": 48, "receiver_seconds": 0.1,
                  "preprocessing_seconds": 0.01, **representation}
        allowed = {("a", 2000, "low")}
        self.assertEqual(verify_cached_records([record], allowed, {("a", 2000): representation}, "r", "x"), allowed)
        for bad in ({**record, "receiver_sha256": "changed"}, {**record, "packet_sha256": "changed"},
                    {**record, "receiver_seconds": float("nan")}, {**record, "actual_visual_tokens": 0}):
            with self.assertRaises(ValueError):
                verify_cached_records([bad], allowed, {("a", 2000): representation}, "r", "x")
        with self.assertRaises(ValueError):
            verify_cached_records([record, record], allowed, {("a", 2000): representation}, "r", "x")


if __name__ == "__main__":
    unittest.main()
