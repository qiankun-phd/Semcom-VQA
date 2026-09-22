"""Fast stdlib-only integrity tests; no model or sealed truth is loaded."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from runtime import (configuration_order, development_rows, freeze_or_verify,
                     read_json, save_json, set_pixel_target, sha256,
                     validate_protocol, validate_resume, verify_representations,
                     verify_tier_separation, visual_measurements)


TIERS = [{"name": n, "pixels": p} for n, p in zip(("low", "medium", "high"), (50176, 100352, 200704))]


class RuntimeTests(unittest.TestCase):
    def test_protocol(self) -> None:
        protocol = {"split": "dev", "expected_questions": 120, "budgets": [2000, 4000, 8000], "tiers": TIERS, "seed": 1, "max_new_tokens": 16}
        budgets, tiers = validate_protocol(protocol)
        self.assertEqual((len(budgets), len(tiers)), (3, 3))
        with self.assertRaises(ValueError):
            validate_protocol({"budgets": [2000], "tiers": TIERS})
        with self.assertRaises(ValueError):
            validate_protocol({**protocol, "split": "test"})

    def test_development_balanced_and_sealed_guard(self) -> None:
        rows = [{"id": f"joint-dev-{i}", "image_id": i, "question_type": str(i % 6)} for i in range(12)]
        ordered = development_rows(rows, 1)
        self.assertEqual(len({r["question_type"] for r in ordered[:6]}), 6)
        self.assertEqual(ordered, development_rows(list(reversed(rows)), 1))
        for bad in ({**rows[0], "id": "joint-test-0"}, {**rows[0], "answer": "yes"}):
            with self.assertRaises(ValueError):
                development_rows([bad, *rows[1:]], 1)

    def test_counterbalanced_order(self) -> None:
        orders = [configuration_order([2000, 4000, 8000], TIERS, 123, i) for i in range(9)]
        for position in range(9):
            self.assertEqual(len({order[position] for order in orders}), 9)

    def test_pixel_controls_and_grid(self) -> None:
        processor = SimpleNamespace(image_processor=SimpleNamespace(size={"shortest_edge": 1, "longest_edge": 2}, merge_size=2, patch_size=16))
        set_pixel_target(processor, 200704)
        self.assertEqual(processor.image_processor.min_pixels, 200704)
        self.assertEqual(processor.image_processor.size["longest_edge"], 200704)
        measured = visual_measurements({"image_grid_thw": [[1, 24, 36]], "input_ids": [[9] * 216 + [3, 4]]}, processor, 9)
        self.assertEqual(measured["actual_visual_tokens"], 216)  # Not 200704 / 1024 = 196.
        self.assertEqual(measured["input_tokens"], 218)
        with self.assertRaises(ValueError):
            visual_measurements({"image_grid_thw": [[1, 24, 36]], "input_ids": [[9] * 196]}, processor, 9)

    def test_tier_separation(self) -> None:
        records = [{"id": "a", "budget": 2000, "tier": t["name"], "actual_visual_tokens": c} for t, c in zip(TIERS, (54, 96, 216))]
        verify_tier_separation(records, TIERS)
        records[1]["actual_visual_tokens"] = 54
        with self.assertRaises(ValueError):
            verify_tier_separation(records, TIERS)

    def test_cross_budget_grid_invariance(self) -> None:
        records = [{"id": "a", "budget": b, "tier": t["name"], "actual_visual_tokens": c,
                    "image_grid_thw": [1, 4, c]} for b in (2000, 4000, 8000) for t, c in zip(TIERS, (54, 96, 216))]
        verify_tier_separation(records, TIERS)
        records[-1]["image_grid_thw"] = [1, 6, 144]
        with self.assertRaises(ValueError):
            verify_tier_separation(records, TIERS)

    def test_resume_guard(self) -> None:
        record = {"id": "a", "budget": 2000, "tier": "low", "receiver_sha256": "r", "protocol_sha256": "p"}
        allowed = {("a", 2000, "low")}
        self.assertEqual(validate_resume([record], allowed, "r", "p"), allowed)
        for records in ([record, record], [{**record, "receiver_sha256": "wrong"}]):
            with self.assertRaises(ValueError):
                validate_resume(records, allowed, "r", "p")

    def test_atomic_freeze_and_packet_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "frozen.json"
            freeze_or_verify(checkpoint, {"hash": "a"})
            self.assertEqual(read_json(checkpoint), {"hash": "a"})
            freeze_or_verify(checkpoint, {"hash": "a"})
            with self.assertRaises(ValueError):
                freeze_or_verify(checkpoint, {"hash": "b"})
            packet, decoded = root / "packet.bin", root / "decoded.png"
            # Test fixtures, not production representations.
            packet.write_bytes(b"packet")
            decoded.write_bytes(b"pixels")
            representation = {"id": "a", "budget": 2000, "image_bytes": 6, "roundtrip": True,
                              "packet": str(packet), "decoded": str(decoded),
                              "packet_sha256": sha256(packet), "decoded_sha256": sha256(decoded)}
            verify_representations([representation], [{"id": "a"}], [2000])
            packet.write_bytes(b"broken")
            with self.assertRaises(ValueError):
                verify_representations([representation], [{"id": "a"}], [2000])


if __name__ == "__main__":
    unittest.main()
