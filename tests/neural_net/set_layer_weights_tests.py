import unittest
import numpy as np
from main import NeuralNet


class SetLayerWeights(unittest.TestCase):
    def test_with_single_hidden_layer(self):
        nnet = NeuralNet(layer_sizes=[1, 2, 2])
        weights = [[3], [-3.5]]
        nnet.set_layer_weights(layer=1, weights=np.array(weights, float))
        self.assertEqual(nnet.weights()[0].tolist(), weights)
        self.assertEqual(nnet.weights()[1].tolist(), [[0, 0], [0, 0]])

        weights2 = [[0, 10], [1, 1]]
        nnet.set_layer_weights(layer=2, weights=np.array(weights2, float))
        self.assertEqual(nnet.weights()[0].tolist(), weights)
        self.assertEqual(nnet.weights()[1].tolist(), weights2)

    def test_with_2_hidden_layers(self):
        nnet = NeuralNet(layer_sizes=[1, 2, 2, 3])
        weights2 = [[0, 10], [1, 1]]
        weights3 = [[0, 10], [1, 1], [-1.5, 2]]

        nnet.set_layer_weights(layer=2, weights=np.array(weights2, float))
        nnet.set_layer_weights(layer=3, weights=np.array(weights3, float))

        self.assertEqual(nnet.weights()[0].tolist(), [[0], [0]])
        self.assertEqual(nnet.weights()[1].tolist(), weights2)
        self.assertEqual(nnet.weights()[2].tolist(), weights3)

    def test_raises_exception_for_array_with_wrong_layer_index(self):
        nnet = NeuralNet(layer_sizes=[1, 2, 2])
        self.assertRaises(NeuralNet.LayerOutOfBound,
                          lambda: nnet.set_layer_weights(layer=-1, weights=[]))
        self.assertRaises(NeuralNet.LayerOutOfBound,
                          lambda: nnet.set_layer_weights(layer=0, weights=[]))
        self.assertRaises(NeuralNet.LayerOutOfBound,
                          lambda: nnet.set_layer_weights(layer=3, weights=[]))
        self.assertRaises(NeuralNet.LayerOutOfBound,
                          lambda: nnet.set_layer_weights(layer=30, weights=[]))

    def test_raises_exception_for_array_with_wrong_dimension(self):
        nnet = NeuralNet(layer_sizes=[1, 2, 2])

        self.assertRaises(
            NeuralNet.InvalidMatrixDimensions,
            lambda: nnet.set_layer_weights(layer=1, weights=np.zeros((2, 2)))
        )

        self.assertRaises(
            NeuralNet.InvalidMatrixDimensions,
            lambda: nnet.set_layer_weights(layer=2, weights=np.zeros((4, 3)))
        )
