import unittest
from main import NeuralNet, NetFactory


class LayerSizesTests(unittest.TestCase):
    def test_layer_sizes(self):
        nnet = NetFactory.create_neural_net(sizes=[8, 24, 15])
        sizes = nnet.layer_sizes()
        self.assertEqual(sizes, [8, 24, 15])

        nnet = NetFactory.create_neural_net(sizes=[2, 3, 18, 8])
        sizes = nnet.layer_sizes()
        self.assertEqual(sizes, [2, 3, 18, 8])
