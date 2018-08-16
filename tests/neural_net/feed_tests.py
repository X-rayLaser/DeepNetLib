import unittest
import numpy as np
from main import NeuralNet
from activation_functions import Rectifier


class NeuralNetFeedforward(unittest.TestCase):
    def test_feed_after_initialization(self):
        nnet = NeuralNet(layer_sizes=[3, 2, 2])
        x = np.array([1, 9, 323], float)
        a = nnet.feed(x)
        self.assertEqual(a[0], 0.5)
        self.assertEqual(a[1], 0.5)

    def test_feed_into_layer(self):
        nnet = NeuralNet(layer_sizes=[2, 3, 2])

        x = np.array([5, 10], float)
        a, z = nnet.feed_into_layer(x, layer=0)

        self.assertTupleEqual(a.shape, (3,))
        self.assertTupleEqual(z.shape, (3,))
        self.assertEqual(z[0], 0)
        self.assertEqual(z[1], 0)
        self.assertEqual(a[0], 0.5)
        self.assertEqual(a[1], 0.5)

        x = np.array([5, 10, 2], float)
        a, z = nnet.feed_into_layer(x, layer=1)
        self.assertTupleEqual(a.shape, (2,))
        self.assertTupleEqual(z.shape, (2,))
        self.assertEqual(z[0], 0)
        self.assertEqual(z[1], 0)
        self.assertEqual(a[0], 0.5)
        self.assertEqual(a[1], 0.5)

    def test_feed_into_with_rectified_unit(self):
        nnet = NeuralNet(layer_sizes=[3, 1, 2])
        nnet.set_output_activation_function(activation=Rectifier)
        nnet.set_layer_biases(layer=2, bias_vector=np.array([-1, 2], float))
        nnet.set_layer_weights(layer=2, weights=np.array([[0.1], [0]], float))

        x = np.array([3], float)
        a, z = nnet.feed_into_layer(x, layer=1)
        self.assertEqual(a[0], 0)
        self.assertEqual(a[1], 2)
