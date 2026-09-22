"""Regression gates for independent routing experiments; no original data needed."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

RUNNER=Path(__file__).parents[1]/"scripts/revision_20260907.py"
if not RUNNER.exists():
    RUNNER=Path(__file__).parent/"revision_20260907.py"
SPEC=importlib.util.spec_from_file_location("revision",RUNNER)
R=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R)


class RevisionTests(unittest.TestCase):
    def fixture(self,split="train",gt=8):
        return {"image":"test-image","question":"How many cars?","snr":5,"qt":"counting",
                "class":"car","raw":4,"split":split,"1":{"correct":False,"transmitted":4,"gt":gt},
                "2":{"correct":False}}

    def test_safe_features_ignore_privileged_fields(self):
        a=self.fixture()
        b=self.fixture(gt=1000)
        b.update(presence_polarity="negative",risk_level="critical",view_quality_bin="good")
        self.assertEqual(R.features(a),R.features(b))
        self.assertEqual(len(R.features(a)),18)

    def test_reject_test_calibration(self):
        with self.assertRaises(ValueError):
            R.fit_calibration([self.fixture("test")])

    def test_count_calibration_frozen(self):
        ratios=R.fit_calibration([self.fixture()])
        self.assertEqual(ratios,{"car|5":2.0})
        self.assertTrue(R.labels([self.fixture("test")],ratios)[0,0])
        self.assertFalse(R.labels([self.fixture("test")],{})[0,0])
        self.assertEqual(ratios,{"car|5":2.0})

    def test_no_calibration_below_three(self):
        self.assertEqual(R.apply_ratio(2,4),2)

    def test_unpriced_and_common_cost_invariance(self):
        p=np.array([[.4,.5],[.6,.5],[.5,.5]])
        e=np.array([[.43,32.5]]*3)
        np.testing.assert_equal(R.choose(p,e,0),[1,0,0])
        np.testing.assert_equal(R.choose(p,e,.01),R.choose(p,e+123,.01))

    def test_detector_always_charged(self):
        g=self.fixture(); g["1"]["bytes"]=1000
        pl={"records":{"test-image|5":{"airtime_s":.3}}}
        fixed=R.energy_matrix([g],pl,32.31,False)
        learned=R.energy_matrix([g],pl,32.31,True)
        self.assertAlmostEqual(learned[0,1]-fixed[0,1],R.ENERGY_DET)
        self.assertEqual(learned[0,0],fixed[0,0])

    def test_validation_lut_forbidden(self):
        with self.assertRaises(ValueError):
            R.fit_lut([self.fixture("validation")],np.array([[0,1]]))

    def test_split_is_image_only(self):
        self.assertEqual(R.split_for("abc"),R.split_for("abc"))
        self.assertEqual(set(R.split_for(str(i)) for i in range(100)),{"train","validation","test"})


if __name__=="__main__":
    unittest.main()
