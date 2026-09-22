import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import numpy as np
import accuracy_upgrade_20260908 as a

class FeatureTests(unittest.TestCase):
    def test_templates(self):
        self.assertEqual(a.parse_question('Are there more car than awning-tricycle objects in this area?','comparison','car'),('awning-tricycle',None))
        self.assertEqual(a.parse_question('Are there at least 12 people objects in this area?','threshold','people'),(None,12))
        self.assertEqual(a.parse_question('Are there both car and bus objects in this area?','co_presence','car'),('bus',None))
    def test_unknown_and_mismatch_rejected(self):
        for text,cls in [('Are there no car objects in this area?','car'),('Are there car objects in this area?','bus')]:
            with self.assertRaises(ValueError): a.parse_question(text,'presence',cls)
    def test_scaler_train_only(self):
        train=np.arange(68).reshape(2,34).astype(float)
        mean,sd=a.scaler(train)
        np.testing.assert_array_equal(mean,np.arange(34)+17)
        x=np.ones((1,18))
        result=a.compose(x,np.zeros((1,34)),'combined',mean,sd)
        np.testing.assert_array_equal(result[:,:18],x)
        self.assertEqual(result.shape,(1,52))
    def test_schema_and_parameters(self):
        self.assertEqual([18+len(v) for v in a.GROUPS.values()],[18,28,42,52])
        self.assertEqual(a.params(a.base.CONFIGS['wide_bce'],18),6658)
        self.assertTrue(all(not any(s in name for s in ('ground_truth','answer','received','correct')) for name in a.QUALITY+a.SEMANTICS))
    def test_base_identity(self):
        x=np.arange(36).reshape(2,18)
        np.testing.assert_array_equal(a.compose(x,np.zeros((2,34)),'base18',np.zeros(34),np.ones(34)),x)
    def test_labels_received_values_never_enter_features(self):
        key={'image':'example','class':'car','qt':'threshold','question':'Are there at least 2 car objects in this area?'}
        image=MagicMock();image.__enter__.return_value.width=100;image.__enter__.return_value.height=100
        boxes={'example':[('car',.8,100),('car',.3,200),('bus',.9,300)]}
        with patch.object(a.Image,'open',return_value=image):
            first=a.enhanced_features([{**key,'ground_truth_answer':'yes','received_count':2,'correct':True}],boxes,Path('/unused'))
            second=a.enhanced_features([{**key,'ground_truth_answer':'no','received_count':200,'correct':False}],boxes,Path('/unused'))
        np.testing.assert_array_equal(first,second)
        self.assertEqual(first[0,-1],1)

if __name__=='__main__': unittest.main()
