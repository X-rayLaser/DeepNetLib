import unittest
import numpy as np
from main import NeuralNet, NetFactory


class SetLayerBiases(unittest.TestCase):
    def test_with_single_hidden_layer(self):
        nnet = NetFactory.create_neural_net(sizes=[1, 2, 3])
        biases1 = [9, -2.5]
        biases2 = [1.5, 0, 5]

        nnet.layers()[0].set_biases(np.array(biases1, float))
        nnet.layers()[1].set_biases(np.array(biases2, float))

        self.assertEqual(nnet.biases()[0].tolist(), biases1)
        self.assertEqual(nnet.biases()[1].tolist(), biases2)

    def test_with_2_hidden_layers(self):
        nnet = NetFactory.create_neural_net(sizes=[1, 2, 2, 1])
        biases2 = [9, -2.5]
        biases3 = [1.5]

        nnet.layers()[1].set_biases(np.array(biases2, float))
        nnet.layers()[2].set_biases(np.array(biases3, float))

        self.assertEqual(nnet.biases()[0].tolist(), [0, 0])
        self.assertEqual(nnet.biases()[1].tolist(), biases2)
        self.assertEqual(nnet.biases()[2].tolist(), biases3)
