"""Synthetic-only tests; no private predictions or development labels loaded."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest

import analyze_grid as ag


def fixture(n: int = 6) -> tuple[list[dict], list[dict], dict]:
    protocol = {"split": "dev", "expected_questions": n, "receiver_sha256": "a" * 64,
                "expected_question_type_counts": {"presence": n}}
    truth = [{"id": f"dev-{index}", "answer": "yes"} for index in range(n)]
    records = []
    for index in range(n):
        for budget, tier in ag.CELLS:
            records.append({"id": f"dev-{index}", "image_id": f"image-{index}", "question_type": "presence",
                            "budget": budget, "tier": tier, "prediction": "YES.",
                            "image_bytes": budget - 10, "actual_visual_tokens": {"low": 49, "medium": 98, "high": 196}[tier],
                            "receiver_seconds": {"low": 1., "medium": 1.5, "high": 2.}[tier],
                            "energy_j": None, "receiver_sha256": "a" * 64, "protocol_sha256": "b" * 64})
    return records, truth, protocol


class AnalysisTests(unittest.TestCase):
    def validate_fixture(self, records=None, truth=None, protocol=None, **kwargs):
        standard = fixture()
        return ag.validate(records if records is not None else standard[0],
                           truth if truth is not None else standard[1],
                           protocol if protocol is not None else standard[2], "b" * 64, **kwargs)

    def test_exact_existing_normalization(self):
        self.assertEqual(ag.normalize("  3. \n"), "three")
        self.assertEqual(ag.normalize("twenty-one"), "twenty-one")
        self.assertEqual(ag.normalize("  RED   CAR!!"), "red car")
        self.assertEqual(ag.normalize("21"), "21")
        self.assertNotEqual(ag.normalize("a cat"), ag.normalize("cat"))

    def test_complete_grid_and_missing_energy(self):
        scored, audit = self.validate_fixture()
        self.assertTrue(audit["decision_eligible"])
        summary, decision = ag.analyze(scored, audit, bootstrap_repeats=40)
        self.assertEqual(summary["cells"]["2000_low"]["ldpc_complex_symbols"]["mean"], 21420)
        self.assertIsNone(summary["cells"]["2000_low"]["energy_j_all_rows_mean"])
        self.assertEqual(summary["cells"]["2000_low"]["energy_missing_count"], 6)
        self.assertEqual(summary["best_fixed_primary_utility"], "2000_low")
        self.assertEqual(summary["independent_fixed_selection"]["cell"], "2000_low")
        self.assertFalse(decision["potential_routing_headroom"])
        self.assertTrue(decision["efficient_fixed_candidate"])
        self.assertTrue(decision["next_stage_requires_frozen_deployable_policy"])
        self.assertFalse(decision["independent_test_authorized_by_this_analysis"])

    def test_complete_required_and_partial_nondecision(self):
        records, truth, protocol = fixture()
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            ag.validate(records[:-1], truth, protocol, "b" * 64)
        scored, audit = ag.validate(records[:-1], truth, protocol, "b" * 64, allow_partial=True)
        self.assertEqual(audit["n_questions"], 5)
        self.assertFalse(audit["decision_eligible"])
        _, decision = ag.analyze(scored, audit, bootstrap_repeats=20)
        self.assertIsNone(decision["efficient_fixed_candidate"])
        self.assertIsNone(decision["potential_routing_headroom"])
        self.assertEqual(decision["recommendation"], "smoke_nondecision")

    def test_duplicate_and_unknown_id_rejected(self):
        records, truth, protocol = fixture()
        with self.assertRaisesRegex(ValueError, "Duplicate prediction"):
            ag.validate(records + [records[0]], truth, protocol, "b" * 64)
        records[0]["id"] = "joint-test-1"
        with self.assertRaisesRegex(ValueError, "outside development"):
            ag.validate(records, truth, protocol, "b" * 64)

    def test_protocol_and_receiver_hashes(self):
        for field in ("protocol_sha256", "receiver_sha256"):
            records, truth, protocol = fixture()
            records[0][field] = "c" * 64
            with self.assertRaisesRegex(ValueError, "hashes"):
                ag.validate(records, truth, protocol, "b" * 64)

    def test_cannot_silently_change_preregistered_criterion(self):
        records, truth, protocol = fixture()
        protocol["primary_lambda"] = .1
        with self.assertRaisesRegex(ValueError, "preregistered"):
            ag.validate(records, truth, protocol, "b" * 64)

    def test_nonfinite_and_resource_guards(self):
        for field, bad in (("receiver_seconds", float("nan")), ("image_bytes", 2001),
                           ("energy_j", float("inf")), ("actual_visual_tokens", 0), ("image_bytes", True)):
            records, truth, protocol = fixture()
            records[0][field] = bad
            with self.subTest(field=field, bad=bad), self.assertRaises(ValueError):
                ag.validate(records, truth, protocol, "b" * 64)

    def test_actual_tiers_must_differ(self):
        records, truth, protocol = fixture()
        for row in records:
            row["actual_visual_tokens"] = 196
        with self.assertRaisesRegex(ValueError, "distinct"):
            ag.validate(records, truth, protocol, "b" * 64)

    def test_same_payload_and_geometry_per_image(self):
        for field in ("image_bytes", "actual_visual_tokens"):
            records, truth, protocol = fixture()
            records[0][field] -= 1
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "inconsistent"):
                ag.validate(records, truth, protocol, "b" * 64)

    def test_sealed_test_and_ambiguous_truth_never_opened(self):
        with tempfile.TemporaryDirectory(prefix="rgb-analysis-") as temp:
            root = Path(temp)
            for name in ("test_truth.sealed.json", "truth.json", "dev_test_truth.json"):
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, "development truth"):
                    ag.read_dev_truth(root / name)
            dev = root / "dev_truth.sealed.json"
            dev.write_text("[]")
            self.assertEqual(ag.read_dev_truth(dev), [])
            test = root / "test_truth.sealed.json"
            test.write_text("[]")
            link = root / "dev_alias.json"
            link.symlink_to(test)
            with self.assertRaisesRegex(ValueError, "development truth"):
                ag.read_dev_truth(link)

    def test_test_split_rejected(self):
        records, truth, protocol = fixture()
        protocol["split"] = "test"
        with self.assertRaisesRegex(ValueError, "development protocol"):
            ag.validate(records, truth, protocol, "b" * 64)

    def test_oracle_not_automatically_gate_success(self):
        records, truth, protocol = fixture(12)
        for row in records:
            # Only compute varies: joint cannot beat compute-only oracle.
            row["image_bytes"] = 1990
            desired = ag.TIERS[int(row["id"].split("-")[1]) % 3]
            row["prediction"] = "yes" if row["tier"] == desired else "no"
        scored, audit = ag.validate(records, truth, protocol, "b" * 64)
        summary, decision = ag.analyze(scored, audit, bootstrap_repeats=20)
        self.assertEqual(summary["oracle_upper_bounds"]["0.05"]["joint_oracle"]["correct"], 12)
        self.assertFalse(decision["potential_routing_headroom"])

    def test_two_axis_headroom_is_only_screen(self):
        records, truth, protocol = fixture(18)
        for row in records:
            index = int(row["id"].split("-")[1])
            desired = ag.CELLS[index % 9]
            row["prediction"] = "yes" if (row["budget"], row["tier"]) == desired else "no"
        scored, audit = ag.validate(records, truth, protocol, "b" * 64)
        _, decision = ag.analyze(scored, audit, bootstrap_repeats=20)
        self.assertTrue(decision["potential_routing_headroom"])
        self.assertFalse(decision["independent_test_authorized_by_this_analysis"])

    def test_fixed_low_plus_adaptive_rate_blocks_joint_gate(self):
        records, truth, protocol = fixture(12)
        for row in records:
            desired_rate = ag.BUDGETS[int(row["id"].split("-")[1]) % 3]
            row["prediction"] = "yes" if row["budget"] == desired_rate and row["tier"] == "low" else "no"
        scored, audit = ag.validate(records, truth, protocol, "b" * 64)
        summary, decision = ag.analyze(scored, audit, bootstrap_repeats=20)
        upper = summary["oracle_upper_bounds"]["0.05"]
        self.assertEqual(upper["rate_only_selected_fixed_tier"], "low")
        self.assertEqual(upper["joint_oracle"]["utility_mean"], upper["rate_only_oracle_best_fixed_tier"]["utility_mean"])
        self.assertFalse(decision["potential_routing_headroom"])

    def test_efficient_fixed_never_increases_image_bytes(self):
        scored, audit = self.validate_fixture()
        _, decision = ag.analyze(scored, audit, bootstrap_repeats=20)
        self.assertTrue(decision["efficient_fixed_candidates"])
        for candidate in decision["efficient_fixed_candidates"]:
            self.assertFalse(candidate["cell"].startswith("8000_"))
            self.assertLessEqual(candidate["mean_image_byte_ratio_vs_reference"], 1.0)

    def test_bootstrap_clusters_duplicate_image_questions(self):
        records, truth, protocol = fixture()
        for row in records:
            row["image_id"] = str(int(row["id"].split("-")[1]) // 2)
        scored, audit = ag.validate(records, truth, protocol, "b" * 64)
        candidate = [row for row in scored if row["budget"] == 2000 and row["tier"] == "low"]
        reference = [row for row in scored if row["budget"] == 4000 and row["tier"] == "high"]
        result = ag.paired_comparison(candidate, reference, repeats=20)
        self.assertEqual(result["n_image_clusters"], 3)
        self.assertEqual(result["n_questions"], 6)
        self.assertEqual(result["paired_image_bootstrap"]["accuracy_difference_percentile_95"], [0., 0.])

    def test_export_is_finite_json_and_report_labeled(self):
        scored, audit = self.validate_fixture()
        summary, decision = ag.analyze(scored, audit, bootstrap_repeats=20)
        json.dumps(summary, allow_nan=False)
        json.dumps(decision, allow_nan=False)
        self.assertIn("not deployable", ag.markdown_report(summary, decision))
        self.assertIn("unavailable", ag.markdown_report(summary, decision))

    def test_cli_writes_four_artifacts_from_jsonl(self):
        records, truth, protocol = fixture(3)
        with tempfile.TemporaryDirectory(prefix="rgb-analysis-") as temp:
            root = Path(temp)
            protocol_path = root / "protocol.json"
            protocol_path.write_text(json.dumps(protocol))
            for row in records:
                row["protocol_sha256"] = ag.digest(protocol_path)
            (root / "records.jsonl").write_text("\n".join(json.dumps(row) for row in records))
            (root / "dev_truth.json").write_text(json.dumps(truth))
            command = [sys.executable, str(Path(ag.__file__)), "--records", str(root / "records.jsonl"),
                       "--truth", str(root / "dev_truth.json"), "--protocol", str(protocol_path),
                       "--output", str(root / "analysis"), "--allow-partial"]
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            self.assertEqual(json.loads(result.stdout)["recommendation"], "smoke_nondecision")
            self.assertEqual({path.name for path in (root / "analysis").iterdir()},
                             {"summary.json", "decision.json", "scored.json", "report.md"})


if __name__ == "__main__":
    unittest.main()
