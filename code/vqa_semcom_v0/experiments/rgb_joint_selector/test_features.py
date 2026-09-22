"""Synthetic tests for pre-decision feature extraction and routing frame."""
from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest

from features import QUESTION_DIM, IMAGE_DIM, question_features, image_features, extract
from route_frame import pack, unpack


class FeatureTests(unittest.TestCase):
    def test_actual_frame_preserves_payload_and_tier(self) -> None:
        for tier in ("low", "medium", "high"):
            payload = b"neural packet bytes"
            frame = pack(payload, tier)
            self.assertEqual(len(frame), len(payload) + 1)
            self.assertEqual(unpack(frame), (tier, payload))
        with self.assertRaises(ValueError):
            unpack(b"\xffbad")
        with self.assertRaises(ValueError):
            pack(b"", "low")

    def test_stable_question_hash(self) -> None:
        first = question_features("How MANY cars are there?")
        self.assertEqual(first, question_features("how many cars are there?"))
        self.assertEqual(len(first), QUESTION_DIM)
        self.assertAlmostEqual(sum(value * value for value in first), 1.0)
        self.assertNotEqual(first, question_features("What color is the car?"))

    def test_empty_question_rejected(self) -> None:
        with self.assertRaises(ValueError):
            question_features(" ")

    def test_synthetic_image_features_and_no_metadata_input(self) -> None:
        try:
            from PIL import Image
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("Pillow/numpy image runtime not installed")
        with tempfile.TemporaryDirectory(prefix="selector-features-") as folder:
            path = Path(folder) / "fixture.png"
            Image.new("RGB", (32, 64), "red").save(path)
            vector = image_features(path)
            self.assertEqual(len(vector), IMAGE_DIM)
            self.assertTrue(all(math.isfinite(value) for value in vector))
            self.assertEqual(vector[-1], .5)
            record = extract("Is there a red object?", path)
            self.assertEqual(set(record), {"question_features", "image_features", "question_feature_seconds", "image_feature_seconds"})
            self.assertGreaterEqual(record["image_feature_seconds"], 0)


if __name__ == "__main__":
    unittest.main()
