from unittest import TestCase

from nn.data_prep import generate_samples
import numpy as np
import itertools as it


class DataPrepTestCase(TestCase):
    def test_generate_samples(self):
        inputs = np.random.rand(200, 13)
        labels = np.random.rand(200, 1)
        gen = generate_samples(40, inputs, labels)
        res = list(it.islice(gen, 10))
        self.assertEqual(10, len(res))
        for i in res:
            self.assertEqual((40, 13), i[0].shape)
            self.assertEqual((40, 1), i[1].shape)
