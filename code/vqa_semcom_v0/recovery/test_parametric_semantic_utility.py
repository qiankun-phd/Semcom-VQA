from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from vqa_semcom.semantic.utility import SemanticUtilityCell, write_semantic_utility_csv
from vqa_semcom.semantic.parametric_utility import (
    BayesianLogistic,
    FeatureScaler,
    ParametricConfig,
    ParametricSemanticUtilityModel,
    encode_key,
)
import numpy as np


SNR_LABELS = ["-5dB", "0dB", "5dB", "10dB", "15dB", "20dB"]
SNR_DB = {"-5dB": -5.0, "0dB": 0.0, "5dB": 5.0, "10dB": 10.0, "15dB": 15.0, "20dB": 20.0}


def _make_cell(q, sl, snr, view, fresh, risk, mean, n):
    # Wilson-ish bounds are not needed for fitting; store mean as raw mean.
    lo = max(0.0, mean - 0.1)
    hi = min(1.0, mean + 0.1)
    return SemanticUtilityCell(
        question_type=q, service_level=sl, channel_bin="medium", snr_bin=snr,
        view_quality_bin=view, freshness_bin=fresh, risk_level=risk,
        sample_count=n, accuracy_mean=mean, accuracy_ci_low=lo, accuracy_ci_high=hi,
        accuracy_lcb=lo, payload_kb=(0.0 if sl == 0 else 5.0 * sl), uncertainty=0.1,
        raw_accuracy_mean=mean, raw_accuracy_ci_low=lo, raw_accuracy_ci_high=hi,
        raw_payload_kb=(0.0 if sl == 0 else 5.0 * sl), payload_bytes=0.0, calibration_note="raw",
    )


def _synthetic_cells(n_per=60):
    """Monotone-in-SNR ground truth so the fit should recover positive slope."""
    cells = []
    for q in ("presence", "counting"):
        for sl in (0, 1, 2):
            for snr in SNR_LABELS:
                for view in ("poor", "medium", "good"):
                    for fresh in ("fresh",):
                        for risk in ("normal",):
                            db = SNR_DB[snr]
                            base = 0.45 + 0.18 * sl  # better service -> better
                            snr_term = 0.0 if sl == 0 else 0.02 * db
                            view_term = 0.05 * {"poor": -1, "medium": 0, "good": 1}[view]
                            p = max(0.05, min(0.95, base + snr_term + view_term))
                            cells.append(_make_cell(q, sl, snr, view, fresh, risk, round(p, 4), n_per))
    return cells


class TestBayesianLogistic(unittest.TestCase):
    def test_recovers_positive_snr_slope(self):
        cells = _synthetic_cells()
        m = ParametricSemanticUtilityModel.fit_from_cells(cells, ParametricConfig(), use_raw=True)
        coefs = m.coefficients()
        # Image/token should show positive SNR sensitivity.
        self.assertGreater(coefs["snr_z:image"] + coefs["snr_z"], 0.0)

    def test_predict_bounds_ordered_and_in_range(self):
        cells = _synthetic_cells()
        m = ParametricSemanticUtilityModel.fit_from_cells(cells, ParametricConfig())
        for q in ("presence", "counting"):
            for sl in (0, 1, 2):
                est = m.U_sem(q, sl, 10.0, "medium", "fresh", "normal")
                self.assertGreaterEqual(est.accuracy_mean, est.accuracy_lcb - 1e-9)
                self.assertTrue(0.0 <= est.accuracy_lcb <= 1.0)
                self.assertTrue(0.0 <= est.accuracy_mean <= 1.0)
                self.assertTrue(0.0 <= est.uncertainty <= 1.0)


class TestParametricModel(unittest.TestCase):
    def setUp(self):
        self.cells = _synthetic_cells()
        self.model = ParametricSemanticUtilityModel.fit_from_cells(self.cells, ParametricConfig())

    def test_off_grid_snr_query(self):
        # 7 dB is not on the {-5,0,5,10,15,20} grid; must still return a value
        est = self.model.U_sem("counting", 2, 7.0, "medium", "fresh", "normal")
        self.assertTrue(0.0 <= est.accuracy_mean <= 1.0)
        # Monotone interpolation: 7 dB between 5 dB and 10 dB predictions for image.
        lo = self.model.U_sem("counting", 2, 5.0, "medium", "fresh", "normal").accuracy_mean
        hi = self.model.U_sem("counting", 2, 10.0, "medium", "fresh", "normal").accuracy_mean
        mid = est.accuracy_mean
        self.assertTrue(min(lo, hi) - 1e-6 <= mid <= max(lo, hi) + 1e-6)

    def test_cache_is_snr_invariant(self):
        a = self.model.U_sem("presence", 0, -5.0, "medium", "fresh", "normal").accuracy_mean
        b = self.model.U_sem("presence", 0, 20.0, "medium", "fresh", "normal").accuracy_mean
        self.assertAlmostEqual(a, b, places=6)

    def test_dropin_get_service_candidates(self):
        obs = {
            "question_type": "counting", "snr_bin": "10dB", "view_quality_bin": "medium",
            "freshness_bin": "fresh", "risk_level": "normal", "epsilon": 0.5,
        }
        cands = self.model.get_service_candidates(obs)
        self.assertTrue(len(cands) >= 2)
        for c in cands:
            self.assertTrue(0.0 <= c.accuracy_lcb <= 1.0)

    def test_json_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            mp = Path(d) / "model.json"
            cp = Path(d) / "cells.csv"
            self.model.save_json(mp)
            write_semantic_utility_csv(self.cells, cp)
            loaded = ParametricSemanticUtilityModel.from_json(mp, cp)
            e1 = self.model.U_sem("counting", 2, 7.0, "medium", "fresh", "normal")
            e2 = loaded.U_sem("counting", 2, 7.0, "medium", "fresh", "normal")
            self.assertAlmostEqual(e1.accuracy_mean, e2.accuracy_mean, places=6)
            self.assertAlmostEqual(e1.accuracy_lcb, e2.accuracy_lcb, places=6)

    def test_sparse_cell_uncertainty_higher(self):
        # A cell type with few samples should not be more certain than a dense one.
        dense = ParametricSemanticUtilityModel.fit_from_cells(_synthetic_cells(n_per=400), ParametricConfig())
        sparse = ParametricSemanticUtilityModel.fit_from_cells(_synthetic_cells(n_per=6), ParametricConfig())
        ed = dense.U_sem("counting", 2, 10.0, "medium", "fresh", "normal")
        es = sparse.U_sem("counting", 2, 10.0, "medium", "fresh", "normal")
        self.assertGreaterEqual(es.uncertainty, ed.uncertainty - 1e-9)


if __name__ == "__main__":
    unittest.main()
