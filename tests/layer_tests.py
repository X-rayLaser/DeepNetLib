import unittest
import numpy as np
from main import Layer
import activation_functions


class LayerInitTests(unittest.TestCase):
    def test_init_empty_layers(self):
        self.assertRaises(
            Layer.BadArchitecture,
            lambda: Layer(size=0, prev_size=0, activation=activation_functions.Sigmoid)
        )

        self.assertRaises(
            Layer.BadArchitecture,
            lambda: Layer(size=1, prev_size=0, activation=activation_functions.Sigmoid)
        )

        self.assertRaises(
            Layer.BadArchitecture,
            lambda: Layer(size=0, prev_size=1, activation=activation_functions.Sigmoid)
        )

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


class LayerSetWeightsTests(unittest.TestCase):
    def test_with_single_hidden_layer(self):
        layer = Layer(size=3, prev_size=2, activation=activation_functions.Sigmoid)
        weights = [[0, 10], [1, 1], [-1, 0.5]]
        layer.set_weights(weights=np.array(weights, float))

        self.assertEqual(layer.weights().tolist(), [[0, 10], [1, 1], [-1, 0.5]])

    def test_raises_exception_for_array_with_wrong_dimension(self):
        layer = Layer(size=5, prev_size=2, activation=activation_functions.Sigmoid)

        self.assertRaises(
            Layer.InvalidMatrixDimensions,
            lambda: layer.set_weights(weights=np.zeros((2, 2)))
        )

        self.assertRaises(
            Layer.InvalidMatrixDimensions,
            lambda: layer.set_weights(weights=np.zeros((4, 3)))
        )


class LayerSetBiasesTests(unittest.TestCase):
    def test_with_single_hidden_layer(self):
        layer = Layer(size=3, prev_size=2, activation=activation_functions.Sigmoid)
        biases = [-2, 0.5, 9]
        layer.set_biases(biases=np.array(biases, float))

        self.assertEqual(layer.biases().tolist(), [-2, 0.5, 9])

    def test_raises_exception_for_array_with_wrong_dimension(self):
        layer = Layer(size=5, prev_size=2, activation=activation_functions.Sigmoid)

        self.assertRaises(
            Layer.InvalidMatrixDimensions,
            lambda: layer.set_biases(biases=np.zeros((2, 2)))
        )

        self.assertRaises(
            Layer.InvalidMatrixDimensions,
            lambda: layer.set_biases(biases=np.zeros((4, 3)))
        )


class LayerSetSingleWeightTests(unittest.TestCase):
    def test_set_weight(self):
        layer = Layer(size=2, prev_size=3, activation=activation_functions.Sigmoid)
        layer.set_weight(row=0, col=1, new_value=2)
        layer.set_weight(row=1, col=2, new_value=-0.5)

        expected_w = np.array(
            [[0, 2, 0],
             [0, 0, -0.5]], float
        )

        self.assertEqual(layer.weights().tolist(), expected_w.tolist())

    def test_raises_exception_for_erroneous_index(self):
        layer = Layer(size=2, prev_size=3, activation=activation_functions.Sigmoid)

        self.assertRaises(IndexError,
                          lambda: layer.set_weight(row=2, col=0, new_value=2)
                          )
        self.assertRaises(IndexError,
                          lambda: layer.set_weight(row=0, col=3, new_value=2)
                          )
        self.assertRaises(IndexError,
                          lambda: layer.set_weight(row=1, col=3, new_value=2)
                          )
        self.assertRaises(IndexError,
                          lambda: layer.set_weight(row=2, col=3, new_value=2)
                          )


class LayerSetSingleBiasTests(unittest.TestCase):
    def test_set_bias(self):
        layer = Layer(size=3, prev_size=2, activation=activation_functions.Sigmoid)
        layer.set_bias(row=0, new_value=2)
        layer.set_bias(row=1, new_value=-0.5)

        expected_b = np.array([2, -0.5, 0], float)

        self.assertEqual(layer.biases().tolist(), expected_b.tolist())

    def test_raises_exception_for_erroneous_index(self):
        layer = Layer(size=3, prev_size=2, activation=activation_functions.Sigmoid)

        self.assertRaises(IndexError,
                          lambda: layer.set_bias(row=3, new_value=2)
                          )
        self.assertRaises(IndexError,
                          lambda: layer.set_bias(row=30, new_value=2)
                          )



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
