"""Small deterministic protocol/model checks; no benchmark test accuracy tuning."""
import unittest
import io
import torch
from literature_baselines_20260908_models import TDeepSCVisDrone, RSVQAHead
from literature_baselines_20260908_questions import tokenize


class ModelTests(unittest.TestCase):
    def test_saved_feature_does_not_include_batch_storage(self):
        batch = torch.zeros(7, 2048, 16, 16, dtype=torch.float16)
        output = io.BytesIO()
        torch.save(batch[0].clone(), output)
        self.assertLess(len(output.getvalue()), 2 * batch[0].numel() * batch.element_size())

    def test_numeric_question_tokens_preserve_distinctions(self):
        self.assertEqual(tokenize('Are there 41 awning-tricycle?'),
                         ['are', 'there', 'forty', 'one', 'awning', 'tricycle'])
        self.assertNotEqual(tokenize('more than 41?'), tokenize('more than 43?'))
        self.assertEqual(tokenize('100 0 1001'), ['one', 'hundred', 'zero', 'one', 'thousand', 'one'])

    def test_unit_complex_power(self):
        torch.manual_seed(41)
        x = torch.randn(4, 3, 16)
        received = TDeepSCVisDrone.rician(x, torch.zeros(4), noiseless=True)
        complex_signal = torch.view_as_complex(received.reshape(4, 24, 2).contiguous())
        torch.testing.assert_close(complex_signal.abs().square().mean(-1), torch.ones(4))

    def test_zero_signal_is_finite(self):
        y = TDeepSCVisDrone.rician(torch.zeros(2, 3, 16), torch.tensor([-5., 20.]))
        self.assertTrue(torch.isfinite(y).all())

    def test_rsvqa_shape_and_gradient(self):
        torch.set_num_threads(2)
        head = RSVQAHead(41)
        logits = head(torch.randn(2, 2048, 16, 16), torch.randn(2, 2400))
        self.assertEqual(logits.shape, (2, 41))
        torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
        self.assertIsNotNone(head.visual_compress.weight.grad)

    def test_channel_realization_is_batch_order_invariant(self):
        x = torch.randn(2, 3, 16)
        snr = torch.tensor([-5., 20.])
        together = TDeepSCVisDrone.rician(x, snr, channel_seeds=[31, 32])
        separate = torch.cat([TDeepSCVisDrone.rician(x[i:i+1], snr[i:i+1], channel_seeds=[31+i]) for i in range(2)])
        torch.testing.assert_close(together, separate)


if __name__ == '__main__':
    unittest.main()
