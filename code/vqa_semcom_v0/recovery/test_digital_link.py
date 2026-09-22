from __future__ import annotations

import unittest

import numpy as np

from vqa_semcom.degradation.digital_link import (
    FadingConfig, LDPCConfig, LinkConfig, sample_power_gain, calibrate_link,
    transmit_image, corrupt_detections, link_config_from_dict, link_config_to_dict,
)


class TestFading(unittest.TestCase):
    def test_unit_average_power(self):
        rng = np.random.default_rng(0)
        for kind, k in [("rayleigh", 0.0), ("rician", 6.0), ("awgn", 0.0)]:
            g = sample_power_gain(kind, k, 20000, rng)
            self.assertAlmostEqual(float(g.mean()), 1.0, delta=0.05)
            self.assertTrue((g >= 0).all())

    def test_rician_less_variance_than_rayleigh(self):
        rng = np.random.default_rng(1)
        ray = sample_power_gain("rayleigh", 0.0, 20000, rng).var()
        ric = sample_power_gain("rician", 10.0, 20000, rng).var()
        self.assertLess(ric, ray)


class TestCalibration(unittest.TestCase):
    def setUp(self):
        self.cfg = LinkConfig(fading=FadingConfig("rician", 6.0),
                              ldpc=LDPCConfig(96, 3, 6, maxiter=30), calib_blocks=200)

    def test_fer_decreases_with_snr(self):
        calib = calibrate_link([-5, 0, 10, 20], self.cfg, seed=0)
        fers = [calib[f"{s}dB"]["fer"] for s in (-5, 0, 10, 20)]
        # monotone non-increasing and bounded
        for a, b in zip(fers, fers[1:]):
            self.assertGreaterEqual(a + 1e-9, b)
        self.assertGreater(fers[0], fers[-1])
        for f in fers:
            self.assertTrue(0.0 <= f <= 1.0)
        self.assertLess(fers[-1], 0.2)   # 20 dB Rician should be clean
        self.assertGreater(fers[0], 0.5)  # -5 dB should be mostly lost


class TestImageTransmit(unittest.TestCase):
    def setUp(self):
        self.cfg = LinkConfig(image_grid=8)
        self.img = (np.random.default_rng(0).integers(0, 255, (256, 256, 3))).astype(np.uint8)

    def test_clean_channel_preserves_image(self):
        calib = {"20dB": {"snr_db": 20.0, "ber": 0.0, "fer": 0.0, "code_rate": 0.5, "k": 50, "n": 96}}
        rng = np.random.default_rng(0)
        out, meta = transmit_image(self.img, 20.0, calib, self.cfg, rng)
        self.assertEqual(out.shape, self.img.shape)
        self.assertEqual(meta["lost"], 0)
        # near-identical (only JPEG quantisation differences)
        self.assertLess(float(np.mean(np.abs(out.astype(int) - self.img.astype(int)))), 12.0)

    def test_loss_fraction_tracks_fer(self):
        calib = {"0dB": {"snr_db": 0.0, "ber": 0.1, "fer": 0.5, "code_rate": 0.5, "k": 50, "n": 96}}
        rng = np.random.default_rng(3)
        out, meta = transmit_image(self.img, 0.0, calib, self.cfg, rng)
        self.assertGreater(meta["lost"], 0)
        self.assertAlmostEqual(meta["loss_frac"], 0.5, delta=0.25)

    def test_deterministic_with_seed(self):
        calib = {"5dB": {"snr_db": 5.0, "ber": 0.05, "fer": 0.3, "code_rate": 0.5, "k": 50, "n": 96}}
        a, _ = transmit_image(self.img, 5.0, calib, self.cfg, np.random.default_rng(7))
        b, _ = transmit_image(self.img, 5.0, calib, self.cfg, np.random.default_rng(7))
        self.assertTrue(np.array_equal(a, b))


class TestDetectionCorruption(unittest.TestCase):
    def _recs(self, n=20):
        return [{"class_id": 3, "label": "car", "x1": 10.0, "y1": 20.0, "x2": 30.0, "y2": 40.0} for _ in range(n)]

    def test_clean_channel_unchanged(self):
        calib = {"20dB": {"snr_db": 20.0, "ber": 0.0, "fer": 0.0, "code_rate": 0.5, "k": 50, "n": 96}}
        out, meta = corrupt_detections(self._recs(), 20.0, calib, LinkConfig(), np.random.default_rng(0))
        self.assertEqual(meta["dropped"], 0)
        self.assertEqual(meta["garbled"], 0)
        self.assertEqual(len(out), 20)

    def test_lossy_channel_drops_or_garbles(self):
        calib = {"-5dB": {"snr_db": -5.0, "ber": 0.3, "fer": 0.8, "code_rate": 0.5, "k": 50, "n": 96}}
        out, meta = corrupt_detections(self._recs(40), -5.0, calib, LinkConfig(), np.random.default_rng(0))
        self.assertGreater(meta["dropped"] + meta["garbled"], 0)
        self.assertLessEqual(len(out), 40)


class TestConfigRoundTrip(unittest.TestCase):
    def test_round_trip(self):
        d = {"fading": {"kind": "rayleigh", "k_factor_db": 0.0},
             "ldpc": {"n": 96, "d_v": 3, "d_c": 6, "maxiter": 40},
             "modulation": "bpsk", "packet_payload_bits": 2048, "calib_blocks": 500, "image_grid": 12}
        cfg = link_config_from_dict(d)
        self.assertEqual(cfg.fading.kind, "rayleigh")
        self.assertEqual(cfg.image_grid, 12)
        back = link_config_to_dict(cfg)
        self.assertEqual(back["ldpc"]["n"], 96)
        self.assertEqual(back["packet_payload_bits"], 2048)


if __name__ == "__main__":
    unittest.main()
