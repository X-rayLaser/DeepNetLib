import unittest
from neural_net import NeuralNet


class NeuralNetInitialization(unittest.TestCase):
    def test_valid_biases_initialization(self):
        nnet = NeuralNet(input_size=15)
        self.assertEqual(len(nnet.layers()), 0)

    def test_with_wrong_input_size(self):
        self.assertRaises(NeuralNet.BadArchitecture, lambda: NeuralNet(input_size=0))
        self.assertRaises(NeuralNet.BadArchitecture, lambda: NeuralNet(input_size=-1))
        self.assertRaises(NeuralNet.BadArchitecture, lambda: NeuralNet(input_size=-10))
