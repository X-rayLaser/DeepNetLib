import unittest
import numpy as np
from main import NeuralNet


class SetLayerBiases(unittest.TestCase):
    def test_with_single_hidden_layer(self):
        nnet = NeuralNet(layer_sizes=[1, 2, 3])
        biases1 = [9, -2.5]
        biases2 = [1.5, 0, 5]

        nnet.set_layer_biases(layer=1, bias_vector=np.array(biases1, float))
        nnet.set_layer_biases(layer=2, bias_vector=np.array(biases2, float))

        self.assertEqual(nnet.biases()[0].tolist(), biases1)
        self.assertEqual(nnet.biases()[1].tolist(), biases2)

    def test_with_2_hidden_layers(self):
        nnet = NeuralNet(layer_sizes=[1, 2, 2, 1])
        biases2 = [9, -2.5]
        biases3 = [1.5]

        nnet.set_layer_biases(layer=2, bias_vector=np.array(biases2, float))
        nnet.set_layer_biases(layer=3, bias_vector=np.array(biases3, float))

        self.assertEqual(nnet.biases()[0].tolist(), [0, 0])
        self.assertEqual(nnet.biases()[1].tolist(), biases2)
        self.assertEqual(nnet.biases()[2].tolist(), biases3)

    def test_raises_exception_for_array_with_wrong_layer_index(self):
        nnet = NeuralNet(layer_sizes=[1, 2, 2])
        self.assertRaises(NeuralNet.LayerOutOfBound,
                          lambda: nnet.set_layer_biases(layer=-1, bias_vector=[]))
        self.assertRaises(NeuralNet.LayerOutOfBound,
                          lambda: nnet.set_layer_biases(layer=0, bias_vector=[]))
        self.assertRaises(NeuralNet.LayerOutOfBound,
                          lambda: nnet.set_layer_biases(layer=3, bias_vector=[]))
        self.assertRaises(NeuralNet.LayerOutOfBound,
                          lambda: nnet.set_layer_biases(layer=30, bias_vector=[]))

    def test_raises_exception_for_array_with_wrong_dimension(self):
        nnet = NeuralNet(layer_sizes=[1, 2, 2])

        self.assertRaises(
            NeuralNet.InvalidMatrixDimensions,
            lambda: nnet.set_layer_biases(layer=1, bias_vector=np.zeros((3,)))
        )

        self.assertRaises(
            NeuralNet.InvalidMatrixDimensions,
            lambda: nnet.set_layer_biases(layer=2, bias_vector=np.zeros((4,)))
        )
