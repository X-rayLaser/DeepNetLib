import unittest
import numpy as np
from main import Layer
import activation_functions


class LayerInitTests(unittest.TestCase):
    def test_weights_is_of_correct_type(self):
        layer = Layer(size=5, prev_size=2, activation=activation_functions.Sigmoid)
        weights = layer.weights()
        self.assertIsInstance(weights, np.ndarray)
        self.assertIn(weights.dtype, [np.float32, np.float64])

    def test_biases_is_of_correct_type(self):
        layer = Layer(size=5, prev_size=2, activation=activation_functions.Sigmoid)
        biases = layer.biases()
        self.assertIsInstance(biases, np.ndarray)
        self.assertIn(biases.dtype, [np.float32, np.float64])

    def test_biases_is_correct_vector(self):
        layer = Layer(size=5, prev_size=2, activation=activation_functions.Sigmoid)
        biases = layer.biases()
        self.assertTupleEqual(biases.shape, (5,))

        layer = Layer(size=2, prev_size=3, activation=activation_functions.Sigmoid)
        biases = layer.biases()
        self.assertTupleEqual(biases.shape, (2,))

    def test_weights_is_correct_matrix(self):
        layer = Layer(size=5, prev_size=2, activation=activation_functions.Sigmoid)
        weights = layer.weights()
        self.assertTupleEqual(weights.shape, (5, 2))

        layer = Layer(size=2, prev_size=3, activation=activation_functions.Sigmoid)
        weights = layer.weights()
        self.assertTupleEqual(weights.shape, (2, 3))


class LayerSetActivationTests(unittest.TestCase):
    def test_sigmoid(self):
        layer = Layer(size=2, prev_size=3, activation=activation_functions.Rectifier)
        layer.set_activation(activation_functions.Sigmoid)
        x = np.array([1, 9, 323], float)
        a, z = layer.feed(x)
        self.assertEqual(a[0], 0.5)
        self.assertEqual(a[1], 0.5)

    def test_rectifier(self):
        layer = Layer(size=2, prev_size=2, activation=activation_functions.Sigmoid)
        layer.set_activation(activation_functions.Rectifier)
        x = np.array([1, 9], float)
        a, z = layer.feed(x)
        self.assertEqual(a[0], 0)
        self.assertEqual(a[1], 0)


class LayerRandomizeTests(unittest.TestCase):
    def test(self):
        layer = Layer(size=1, prev_size=3, activation=activation_functions.Sigmoid)
        layer.randomize()
        x = np.array([1, 9, -2], float)
        a, z = layer.feed(x)
        self.assertNotEqual(a[0], 0)


class LayerFeedTests(unittest.TestCase):
    def test_feed_with_sigmoid(self):
        layer = Layer(size=2, prev_size=3, activation=activation_functions.Sigmoid)
        x = np.array([1, 9, 323], float)
        a, z = layer.feed(x)
        self.assertEqual(z[0], 0)
        self.assertEqual(z[1], 0)
        self.assertEqual(a[0], 0.5)
        self.assertEqual(a[1], 0.5)
