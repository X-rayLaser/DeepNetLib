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


class HelpersTests(unittest.TestCase):
    def test_zero_gradients_list(self):
        nnet = NeuralNet(layer_sizes=[3, 5, 4, 1])
        weights_grads, biases_grads = backprop.zero_gradients_list(neural_net=nnet)
        nmatrices = len(weights_grads)
        self.assertEqual(nmatrices, 3)
        self.assertEqual(nmatrices, len(biases_grads))

        self.assertTupleEqual(weights_grads[0].shape, (5, 3))
        self.assertTupleEqual(weights_grads[1].shape, (4, 5))
        self.assertTupleEqual(weights_grads[2].shape, (1, 4))

        self.assertTupleEqual(biases_grads[0].shape, (5,))
        self.assertTupleEqual(biases_grads[1].shape, (4,))
        self.assertTupleEqual(biases_grads[2].shape, (1,))

        for w in weights_grads:
            self.assertAlmostEqual(w.sum(), 0, places=8)

        for b in biases_grads:
            self.assertAlmostEqual(b.sum(), 0, places=8)

    def test_update_total_gradients(self):
        grad_total = [np.array([[100, 120, 130], [50, 10, 60]], int),
                      np.array([[10, 20], [10, 10]], int)]

        grad_last = [np.array([[5, 10, 20], [10, 10, 10]], int),
                     np.array([[1, 1], [2, 4]], int)]

        grad = backprop.update_total_gradients(summed_gradients_list=grad_total,
                                              new_gradients_list=grad_last)

        expected_grad = [np.array([[105, 130, 150], [60, 20, 70]], int),
                         np.array([[11, 21], [12, 14]], int)]

        self.assertEqual(len(grad), 2)
        self.assertTupleEqual(grad[0].shape, (2, 3))
        self.assertTupleEqual(grad[1].shape, (2, 2))

        self.assertTrue(np.all(grad[0] == expected_grad[0]))
        self.assertTrue(np.all(grad[1] == expected_grad[1]))

    def test_average_gradient(self):
        mtx1 = np.array(
            [[5, 10],
             [2, 4]], float
        )

        mtx2 = np.array([[1, 2, 4]], float)
        gradient_sum = [mtx1, mtx2]

        mtx1_expected = np.array(
            [[1, 2],
             [0.4, 0.8]], float
        )

        mtx2_expected = np.array([[0.2, 0.4, 0.8]], float)

        expected_gradient = [mtx1_expected, mtx2_expected]
        grad = backprop.average_gradient(gradient_sum=gradient_sum, examples_count=5)

        self.assertEqual(len(grad), 2)
        self.assertTupleEqual(grad[0].shape, (2, 2))
        self.assertTupleEqual(grad[1].shape, (1, 3))

        self.assertTrue(np.all(grad[0] == expected_gradient[0]))
        self.assertTrue(np.all(grad[1] == expected_gradient[1]))

    def test_get_final_layer_error_for_1_element_vectors(self):
        z_last = np.array([3], float)
        y = np.array([0], float)
        a_last = sigma(z_last)
        nabla = backprop.get_final_layer_error(a_last, y, z_last)
        self.assertAlmostEqual(nabla[0], 0.04, places=2)

        z_last = np.array([-1], float)
        y = np.array([0.5], float)
        a_last = sigma(z_last)
        nabla = backprop.get_final_layer_error(a_last, y, z_last)
        self.assertAlmostEqual(nabla[0], (a_last - y) * sigma_prime(z_last), places=2)

    def test_get_final_layer_error_for_arrays(self):
        z_last = np.array([3, -1], float)
        y = np.array([0, 0.5], float)
        a_last = sigma(z_last)
        nabla = backprop.get_final_layer_error(a_last, y, z_last)

        self.assertAlmostEqual(nabla[0], 0.04, places=2)
        self.assertAlmostEqual(nabla[1], (a_last[1] - y[1]) * sigma_prime(z_last[1]),
                               places=2)

    def test_get_weights_gradient(self):
        layer_error = np.array([3, 5, 10], float)
        activations = np.array([0.5, 0.3], float)
        grad = backprop.get_weights_gradient(layer_error=layer_error,
                                            previous_layer_activations=activations)

        expected_grad = np.array([[1.5, 0.9], [2.5, 1.5], [5., 3.]], float)

        self.assertTupleEqual(grad.shape, expected_grad.shape)
        self.assertTrue(np.allclose(grad, expected_grad))

    def test_get_bias_gradient(self):
        layer_error = np.array([2, 9, 12, 83])
        grad = backprop.get_bias_gradient(layer_error=layer_error)
        self.assertTrue(np.allclose(layer_error, grad))

    def test_get_error_in_layer(self):
        nabla_next = np.array([2, 9, 5], float)
        w_next = np.array([[3, 0], [0, 1], [4, 5]], float)
        z = np.array([2, -1])
        nabla = backprop.get_error_in_layer(nabla_next, w_next, z)

        expected_nabla = np.array([2.72983, 6.6848])
        self.assertTrue(np.allclose(nabla, expected_nabla))

    def test_compute_activations_and_zsums(self):
        nnet = NeuralNet(layer_sizes=[2, 3, 2])
        x = np.array([0.5, 3], float)
        nnet.randomize_parameters()

        a, zs = backprop.compute_activations_and_zsums(x=x, neural_net=nnet)
        expected_activations = nnet.feed(x=x)
        self.assertTrue(np.allclose(a[-1], expected_activations))

    def test_compute_errors(self):
        nnet = NeuralNet(layer_sizes=[1, 2, 1])
        nnet.set_weight(layer=1, row=0, col=0, new_value=-0.5)
        nnet.set_weight(layer=1, row=1, col=0, new_value=1.5)
        nnet.set_weight(layer=2, row=0, col=1, new_value=5)

        nnet.set_bias(layer=1, row=0, new_value=1)
        nnet.set_bias(layer=2, row=0, new_value=-1)

        x = np.array([2], float)

        y = np.array([1], float)

        a, zs = backprop.compute_activations_and_zsums(x=x, neural_net=nnet)

        errors_list = backprop.compute_errors(neural_net=nnet, output_activations=a[-1],
                                             expected_output=y, weighed_sums=zs)

        expected_nabla2 = (a[-1] - y) * sigma_prime(zs[-1])
        expected_nabla1 = backprop.get_error_in_layer(nabla_next=expected_nabla2,
                                                     w_next=np.array([[0, 5]]), z=zs[0])

        self.assertTrue(np.allclose(errors_list[0], expected_nabla1))
        self.assertTrue(np.allclose(errors_list[1], expected_nabla2))


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


if __name__ == '__main__':
    unittest.main()
