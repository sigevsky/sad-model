import unittest
import torch
from nn.sda_nn import SdaNet


class SdaNetTestCase(unittest.TestCase):

    def test_should_yield_valid_dimension_shape(self):
        mfcc_feature_size = 13
        hidden_size = 10
        seq_length = 20
        batch_size = 12

        net = SdaNet(mfcc_feature_size, hidden_size)
        input = torch.rand(batch_size, seq_length, mfcc_feature_size)
        output = net(input)
        self.assertEqual((batch_size, seq_length, 1), output.shape)


if __name__ == '__main__':
    unittest.main()
