import unittest
from detector_decoder_controls_20260908_decoder import packet,predict

class ReceiverControls(unittest.TestCase):
    def test_missing_packet_rejected(self):
        with self.assertRaises(ValueError):packet('sender_count: 99')
    def test_second_class_is_received(self):
        g={'qt':'comparison','class':'car','question':'Are there more car than bus objects in this area?','snr':0}
        self.assertEqual(predict(g,packet('detector_counts_by_class: bus:4, car:2'),{},'baseline'),'no')
    def test_zero_not_invented(self):
        g={'qt':'presence','class':'car','question':'Are there car objects in this area?','snr':0}
        self.assertEqual(predict(g,{}, {'car|0':{'slope':2.,'intercept':10.}},'linear'),'no')
    def test_baseline_only_calibrates_counting(self):
        g={'qt':'threshold','class':'car','question':'Are there at least 5 car objects in this area?','snr':0}
        self.assertEqual(predict(g,{'car':3},{'car|0':2.},'baseline'),'no')
        self.assertEqual(predict(g,{'car':3},{'car|0':{'slope':2.,'intercept':0.}},'linear'),'yes')

if __name__=='__main__':unittest.main()
