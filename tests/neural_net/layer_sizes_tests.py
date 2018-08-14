import unittest
from main import NeuralNet


class LayerSizesTests(unittest.TestCase):
    def test_layer_sizes(self):
        nnet = NeuralNet(layer_sizes=[8, 24, 15])
        sizes = nnet.layer_sizes()
        self.assertEqual(sizes, [8, 24, 15])

        nnet = NeuralNet(layer_sizes=[2, 3, 18, 8])
        sizes = nnet.layer_sizes()
        self.assertEqual(sizes, [2, 3, 18, 8])
