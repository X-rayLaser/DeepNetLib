import unittest
import numpy as np
from main import NeuralNet, GradientDescent
from activation_functions import sigma, sigma_prime
from cost_functions import quadratic_per_example, quadratic_cost
from backprop import back_propagation
import helpers
import backprop
import backprop_slow


class NeuralNetInitialization(unittest.TestCase):
    def test_init_with_too_little_layers(self):
        self.assertRaises(NeuralNet.BadArchitecture, lambda: NeuralNet(layer_sizes=[]))
        self.assertRaises(NeuralNet.BadArchitecture, lambda: NeuralNet(layer_sizes=[2]))
        self.assertRaises(NeuralNet.BadArchitecture, lambda: NeuralNet(layer_sizes=[2, 1]))

    def test_init_empty_layers(self):
        self.assertRaises(NeuralNet.BadArchitecture, lambda: NeuralNet(layer_sizes=[0, 1, 1]))
        self.assertRaises(NeuralNet.BadArchitecture, lambda: NeuralNet(layer_sizes=[13, 0, 1]))
        self.assertRaises(NeuralNet.BadArchitecture, lambda: NeuralNet(layer_sizes=[0, 0, 0]))

    def test_valid_init(self):
        NeuralNet(layer_sizes=[1, 1, 1])
        NeuralNet(layer_sizes=[3, 1, 11, 151, 1])

    def test_valid_weights_initialization(self):
        nnet = NeuralNet(layer_sizes=[10, 4, 3])
        w = nnet.weights()
        self.assertEqual(len(w), 2)
        self.assertEqual(w[0].shape, (4, 10))
        self.assertEqual(w[1].shape, (3, 4))

        nnet = NeuralNet(layer_sizes=[100, 25, 15, 10])
        w = nnet.weights()
        self.assertEqual(len(w), 3)
        self.assertEqual(w[0].shape, (25, 100))
        self.assertEqual(w[1].shape, (15, 25))
        self.assertEqual(w[2].shape, (10, 15))

    def test_valid_biases_initialization(self):
        nnet = NeuralNet(layer_sizes=[10, 4, 3])
        b = nnet.biases()
        self.assertEqual(len(b), 2)
        self.assertEqual(b[0].shape, (4,))
        self.assertEqual(b[1].shape, (3,))

        nnet = NeuralNet(layer_sizes=[25, 4, 15, 5])
        b = nnet.biases()
        self.assertEqual(len(b), 3)
        self.assertEqual(b[0].shape, (4,))
        self.assertEqual(b[1].shape, (15,))
        self.assertEqual(b[2].shape, (5,))


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


class NeuralNetTrain(unittest.TestCase):
    def test_gives_correct_output_on_training_data(self):
        nnet = NeuralNet(layer_sizes=[1, 1, 1])

        xes = [np.array([-10], float), np.array([100], float)]
        ys = [np.array([0.5], float), np.array([0.75], float)]

        nnet.train(examples=(xes, ys), nepochs=100)

        for i in range(len(xes)):
            res = nnet.feed(xes[i])
            self.assertAlmostEqual(res[0], ys[i][0], places=1)

    def test_gives_correct_output_for_unseen_data(self):
        nnet = NeuralNet(layer_sizes=[1, 10, 1])

        def parabola(x):
            return x**2

        examples = helpers.generate_data(f=parabola, start_value=-0.6,
                                         end_value=-0.4, step_value=0.005)

        nnet.train(examples=examples, nepochs=10)

        xval = -0.5000125
        yval = parabola(xval)

        net_input = np.array([xval], float)
        output = nnet.feed(net_input)
        self.assertAlmostEqual(output[0], yval, places=1)


class SetWeight(unittest.TestCase):
    def test_set_weight(self):
        nnet = NeuralNet(layer_sizes=[2, 3, 2])
        nnet.set_weight(layer=1, row=0, col=0, new_value=5)
        nnet.set_weight(layer=1, row=1, col=1, new_value=-2.5)
        nnet.set_weight(layer=2, row=1, col=2, new_value=1.5)

        w = nnet.weights()
        expected_w1 = np.array(
            [[5, 0],
             [0, -2.5],
             [0, 0]], float
        )
        expected_w2 = np.array(
            [[0, 0, 0],
             [0, 0, 1.5]], float
        )

        self.assertTrue(np.allclose(w[0], expected_w1))
        self.assertTrue(np.allclose(w[1], expected_w2))

    def test_raises_exception_for_erroneous_layer(self):
        nnet = NeuralNet(layer_sizes=[2, 1, 2])
        self.assertRaises(NeuralNet.LayerOutOfBound,
                          lambda: nnet.set_weight(layer=0, row=0, col=0, new_value=2)
                          )
        self.assertRaises(NeuralNet.LayerOutOfBound,
                          lambda: nnet.set_weight(layer=-1, row=0, col=0, new_value=2)
                          )
        self.assertRaises(NeuralNet.LayerOutOfBound,
                          lambda: nnet.set_weight(layer=-50, row=0, col=0, new_value=2)
                          )
        self.assertRaises(NeuralNet.LayerOutOfBound,
                          lambda: nnet.set_weight(layer=3, row=0, col=0, new_value=2)
                          )
        self.assertRaises(NeuralNet.LayerOutOfBound,
                          lambda: nnet.set_weight(layer=30, row=0, col=0, new_value=2)
                          )

    def test_raises_exception_for_erroneous_index(self):
        nnet = NeuralNet(layer_sizes=[2, 1, 2])
        self.assertRaises(IndexError,
                          lambda: nnet.set_weight(layer=1, row=1, col=0, new_value=2)
                          )
        self.assertRaises(IndexError,
                          lambda: nnet.set_weight(layer=1, row=0, col=2, new_value=2)
                          )
        self.assertRaises(IndexError,
                          lambda: nnet.set_weight(layer=1, row=1, col=2, new_value=2)
                          )
        self.assertRaises(IndexError,
                          lambda: nnet.set_weight(layer=2, row=2, col=0, new_value=2)
                          )
        self.assertRaises(IndexError,
                          lambda: nnet.set_weight(layer=2, row=1, col=1, new_value=2)
                          )


class SetBias(unittest.TestCase):
    def test_set_bias(self):
        nnet = NeuralNet(layer_sizes=[2, 3, 2])
        nnet.set_bias(layer=1, row=2, new_value=3.3)
        nnet.set_bias(layer=1, row=1, new_value=-1)
        nnet.set_bias(layer=2, row=0, new_value=2)

        expected_bias1 = np.array([0, -1, 3.3], float)
        expected_bias2 = np.array([2, 0], float)

        b = nnet.biases()
        self.assertTrue(np.allclose(b[0], expected_bias1))
        self.assertTrue(np.allclose(b[1], expected_bias2))

    def test_raises_exception_for_erroneous_layer(self):
        nnet = NeuralNet(layer_sizes=[2, 3, 2, 4])
        self.assertRaises(NeuralNet.LayerOutOfBound,
                          lambda: nnet.set_bias(layer=0, row=1, new_value=2)
                          )

        self.assertRaises(NeuralNet.LayerOutOfBound,
                          lambda: nnet.set_bias(layer=4, row=1, new_value=2)
                          )
        self.assertRaises(NeuralNet.LayerOutOfBound,
                          lambda: nnet.set_bias(layer=14, row=1, new_value=2)
                          )

    def test_raises_exception_for_erroneous_index(self):
        nnet = NeuralNet(layer_sizes=[2, 1, 2])
        self.assertRaises(IndexError,
                          lambda: nnet.set_bias(layer=1, row=1, new_value=2)
                          )
        self.assertRaises(IndexError,
                          lambda: nnet.set_bias(layer=2, row=2, new_value=2)
                          )


from tests.activation_functions_tests import *
from tests.cost_functions_tests import *
from tests.gradient_descent_tests import *
from tests.back_propagation_tests import *
from tests.helper_tests import *


if __name__ == '__main__':
    unittest.main()
