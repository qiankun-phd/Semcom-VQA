"""Synthetic validation of the policy-freeze boundary; no sealed data."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import analyze_grid as analysis
from prepare_stage3 import choose_fixed, digest
from test_analysis import fixture


class Stage3Tests(unittest.TestCase):
    def setUp(self) -> None:
        records, truth, protocol = fixture()
        scored, audit = analysis.validate(records, truth, protocol, "b" * 64)
        self.summary, self.decision = analysis.analyze(scored, audit, bootstrap_repeats=20)

    def test_select_simple_fixed_not_oracle(self) -> None:
        self.assertEqual(choose_fixed(self.summary, self.decision), "2000_low")

    def test_incomplete_screen_cannot_freeze(self) -> None:
        self.decision["decision_eligible"] = False
        with self.assertRaisesRegex(ValueError, "complete"):
            choose_fixed(self.summary, self.decision)

    def test_oracle_alone_is_not_deployable(self) -> None:
        self.decision["efficient_fixed_candidate"] = False
        self.decision["potential_routing_headroom"] = True
        self.assertIsNone(choose_fixed(self.summary, self.decision))

    def test_byte_growth_is_rejected(self) -> None:
        self.decision["efficient_fixed_candidates"] = [{"cell": "8000_low"}]
        with self.assertRaisesRegex(ValueError, "screen"):
            choose_fixed(self.summary, self.decision)

    def test_cli_freezes_and_refuses_altered_outputs(self) -> None:
        with tempfile.TemporaryDirectory(prefix="rgb-fixed-freeze-") as folder:
            root = Path(folder)
            records, truth, protocol = fixture()
            protocol["stage3"] = {"test_opened": False}
            protocol_file = root / "protocol.json"
            protocol_file.write_text(json.dumps(protocol))
            for row in records:
                row["protocol_sha256"] = digest(protocol_file)
            records_file = root / "records.json"
            records_file.write_text(json.dumps(records))
            scored, audit = analysis.validate(records, truth, protocol, digest(protocol_file))
            summary, decision = analysis.analyze(scored, audit)
            for name, value in (("summary.json", summary), ("decision.json", decision), ("scored.json", scored)):
                (root / name).write_text(json.dumps(value))
            target = root / "frozen.json"
            command = [sys.executable, str(Path(__file__).with_name("prepare_stage3.py")),
                       "--analysis", str(root), "--records", str(records_file),
                       "--protocol", str(protocol_file), "--output", str(target)]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            frozen = json.loads(target.read_text())
            self.assertEqual(frozen["selected_cell"], "2000_low")
            self.assertFalse(frozen["test_opened"])
            self.assertFalse(frozen["wireless_evaluated"])
            summary["cells"]["2000_low"]["correct"] = 0
            (root / "summary.json").write_text(json.dumps(summary))
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("no longer matches", result.stderr)


if __name__ == "__main__":
    unittest.main()
