import unittest
import numpy as np
from main import NeuralNet, NetFactory


class SetLayerWeights(unittest.TestCase):
    def test_with_single_hidden_layer(self):
        nnet = NetFactory.create_neural_net(sizes=[1, 2, 2])
        weights = [[3], [-3.5]]
        nnet.layers()[0].set_weights(np.array(weights, float))
        self.assertEqual(nnet.weights()[0].tolist(), weights)
        self.assertEqual(nnet.weights()[1].tolist(), [[0, 0], [0, 0]])

        weights2 = [[0, 10], [1, 1]]
        nnet.layers()[1].set_weights(np.array(weights2, float))
        self.assertEqual(nnet.weights()[0].tolist(), weights)
        self.assertEqual(nnet.weights()[1].tolist(), weights2)

    def test_with_2_hidden_layers(self):
        nnet = NetFactory.create_neural_net(sizes=[1, 2, 2, 3])
        weights2 = [[0, 10], [1, 1]]
        weights3 = [[0, 10], [1, 1], [-1.5, 2]]

        nnet.layers()[1].set_weights(np.array(weights2, float))
        nnet.layers()[2].set_weights(np.array(weights3, float))

        self.assertEqual(nnet.weights()[0].tolist(), [[0], [0]])
        self.assertEqual(nnet.weights()[1].tolist(), weights2)
        self.assertEqual(nnet.weights()[2].tolist(), weights3)
