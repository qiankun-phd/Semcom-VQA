"""Artifact smoke test using explicit synthetic fixtures, never reported as data."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

if __package__:
    from .build_report import build_report
    from .evaluate_factorial import GROUPS, evaluate_split, holm_adjust
    from .test_evaluate_factorial import fixture
else:
    from build_report import build_report
    from evaluate_factorial import GROUPS, evaluate_split, holm_adjust
    from test_evaluate_factorial import fixture


class ReportTest(unittest.TestCase):
    def test_real_figure_and_text_artifacts_from_explicit_fixture(self) -> None:
        splits = {split: evaluate_split(*fixture())[0] for split in ("validation", "legacy_dev")}
        tests = [result for split in splits.values() for family in split["factorial_contrasts"].values() for result in family.values()]
        holm_adjust(tests)
        evaluation = {"experiment_id": "EXP-013", "groups": list(GROUPS), "splits": splits,
                      "statistics": {"holm_family_size": 12}, "full_cost_gate": "PENDING",
                      "reproduction": {"passed": True, "max_abs_difference": 0., "action_mismatches": 0},
                      "limitations": ["Synthetic unit fixture; not experimental results."]}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            manifest = build_report(evaluation, output)
            self.assertEqual(len(manifest["figures"]), 4)
            for name in manifest["figures"]:
                path = output / name
                self.assertGreater(path.stat().st_size, 1000)
                self.assertTrue(path.read_bytes().startswith(b"%PDF" if path.suffix == ".pdf" else b"\x89PNG"))
            for name in ("analysis-report.md", "stats-appendix.md", "figure-catalog.md"):
                text = (output / name).read_text()
                self.assertIn("EXP-013" if name == "analysis-report.md" else "Figure" if name == "figure-catalog.md" else "Statistical", text)
            appendix = (output / "stats-appendix.md").read_text()
            self.assertIn("Seed 27", appendix)
            self.assertIn("Holm", appendix)
            self.assertIn("510", (output / "analysis-report.md").read_text())


if __name__ == "__main__":
    unittest.main()
