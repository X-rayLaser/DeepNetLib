import unittest
from main import NeuralNet, Layer, NetFactory


class AddLayerTests(unittest.TestCase):
    def test_layer_sizes_with_add_layer(self):
        nnet = NetFactory.create_neural_net(sizes=[8, 24, 15])

        layer = Layer(size=4, prev_size=15, activation=[])
        nnet.add_layer(layer)
        sizes = nnet.layer_sizes()
        self.assertEqual(sizes, [8, 24, 15, 4])

    def test_add_layer_with_wrong_size(self):
        nnet = NetFactory.create_neural_net(sizes=[8, 2, 3])
        layer = Layer(size=4, prev_size=5, activation=[])

        self.assertRaises(NeuralNet.BadArchitecture, lambda: nnet.add_layer(layer))
