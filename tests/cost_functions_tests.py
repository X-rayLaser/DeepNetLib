import unittest
import numpy as np
import cost_functions
from main import NeuralNet
from activation_functions import sigma, sigma_prime, Sigmoid, Rectifier


class QuadraticCostTests(unittest.TestCase):
    def test_quadratic_cost(self):
        activations = [np.array([0.7, 0.6], float), np.array([1, 0], float)]
        outputs = [np.array([0.2, 0.5], float), np.array([0, 0], float)]
        quadracost = cost_functions.QuadraticCost()
        c = quadracost.compute_cost(activations=activations, outputs=outputs)
        self.assertAlmostEqual(c, 0.315, places=3)

    def test_quadratic_per_example(self):
        a = np.array([0.5, 0.7], float)
        y = np.array([0.2, 0.1], float)
        c = cost_functions.quadratic_per_example(activation=a, expected_output=y)
        self.assertAlmostEqual(c, 0.225, places=3)

        a = np.array([1, 0], float)
        y = np.array([0, 1], float)
        c = cost_functions.quadratic_per_example(activation=a, expected_output=y)
        self.assertAlmostEqual(c, 1, places=1)

    def test_get_cost_initial(self):
        nnet = NeuralNet(layer_sizes=[1, 1, 1])

        xes = [np.array([-10], float), np.array([100], float)]
        ys = [np.array([0.5], float), np.array([0.75], float)]

        examples = (xes, ys)
        cost = nnet.get_cost(examples)
        self.assertAlmostEqual(cost, 1.0/64, places=4)


class QuadraticCostGradientsTests(unittest.TestCase):
    def test_get_final_layer_error_for_1_element_vectors(self):
        quadratic = cost_functions.QuadraticCost()
        z_last = np.array([3], float)
        y = np.array([0], float)
        a_last = sigma(z_last)
        nabla = quadratic.get_final_layer_error(a_last, y, z_last, activation_function=Sigmoid)
        self.assertAlmostEqual(nabla[0], 0.04, places=2)

        z_last = np.array([-1], float)
        y = np.array([0.5], float)
        a_last = sigma(z_last)
        nabla = quadratic.get_final_layer_error(a_last, y, z_last, activation_function=Sigmoid)
        self.assertAlmostEqual(nabla[0], (a_last - y) * sigma_prime(z_last), places=2)

    def test_get_final_layer_error_for_arrays(self):
        quadratic = cost_functions.QuadraticCost()

        z_last = np.array([3, -1], float)
        y = np.array([0, 0.5], float)
        a_last = sigma(z_last)
        nabla = quadratic.get_final_layer_error(a_last, y, z_last, activation_function=Sigmoid)

        self.assertAlmostEqual(nabla[0], 0.04, places=2)
        self.assertAlmostEqual(nabla[1], (a_last[1] - y[1]) * sigma_prime(z_last[1]),
                               places=2)

    def test_get_final_layer_error_for_rectifier_activation(self):
        quadratic = cost_functions.QuadraticCost()
        z_last = np.array([-1], float)
        y = np.array([0.5], float)
        a_last = sigma(z_last)
        nabla = quadratic.get_final_layer_error(a_last, y, z_last, activation_function=Rectifier)
        self.assertAlmostEqual(nabla[0], (a_last - y) * Rectifier.gradient(z_last), places=2)

    def test_get_weights_gradient(self):
        quadratic = cost_functions.QuadraticCost()

        layer_error = np.array([3, 5, 10], float)
        activations = np.array([0.5, 0.3], float)
        grad = quadratic.get_weights_gradient(layer_error=layer_error,
                                              previous_layer_activations=activations)

        expected_grad = np.array([[1.5, 0.9], [2.5, 1.5], [5., 3.]], float)

        self.assertTupleEqual(grad.shape, expected_grad.shape)
        self.assertTrue(np.allclose(grad, expected_grad))

    def test_get_bias_gradient(self):
        quadratic = cost_functions.QuadraticCost()

        layer_error = np.array([2, 9, 12, 83])
        grad = quadratic.get_bias_gradient(layer_error=layer_error)
        self.assertTrue(np.allclose(layer_error, grad))

    def test_get_error_in_layer(self):
        quadratic = cost_functions.QuadraticCost()

        nabla_next = np.array([2, 9, 5], float)
        w_next = np.array([[3, 0], [0, 1], [4, 5]], float)
        z = np.array([2, -1])
        nabla = quadratic.get_error_in_layer(nabla_next, w_next, z, activation_function=Sigmoid)

        expected_nabla = np.array([2.72983, 6.6848])
        self.assertTrue(np.allclose(nabla, expected_nabla))

    def test_get_error_in_layer_using_rectified_activation(self):
        quadratic = cost_functions.QuadraticCost()

        nabla_next = np.array([2, 9, 5], float)
        w_next = np.array([[3, 0], [0, 1], [4, 5]], float)
        z = np.array([-0.01, -10])
        nabla = quadratic.get_error_in_layer(nabla_next, w_next, z, activation_function=Rectifier)

        expected_nabla = np.array([0, 0])
        self.assertTrue(np.allclose(nabla, expected_nabla))


class CrossEntropyCostTests(unittest.TestCase):
    def cross_entropy_cost(self, activations, outputs):
        cost_func = cost_functions.CrossEntropyCost()
        return cost_func.compute_cost(activations=activations, outputs=outputs)

    def test_single_example_1_output_neuron_no_entropy(self):
        a = [np.array([0], float)]
        y = [np.array([0], float)]
        c = self.cross_entropy_cost(activations=a, outputs=y)
        self.assertAlmostEqual(c, 0, places=4)

        a = [np.array([1], float)]
        y = [np.array([1], float)]
        c = self.cross_entropy_cost(activations=a, outputs=y)
        self.assertAlmostEqual(c, 0, places=4)

        a = [np.array([0.5], float)]
        y = [np.array([0.5], float)]
        c = self.cross_entropy_cost(activations=a, outputs=y)
        self.assertAlmostEqual(c, 0.6931, places=4)

    def test_single_example_1_output_neuron_max_entropy(self):
        from math import exp
        a = [np.array([1 - exp(-10)], float)]
        y = [np.array([0], float)]
        c = self.cross_entropy_cost(activations=a, outputs=y)
        self.assertAlmostEqual(c, 10, places=4)

        a = [np.array([exp(-10)], float)]
        y = [np.array([1], float)]
        c = self.cross_entropy_cost(activations=a, outputs=y)
        self.assertAlmostEqual(c, 10, places=4)

    def test_single_example_1_output_neuron_entropy_increases(self):
        a = [np.array([0.1], float)]
        y = [np.array([0], float)]
        c1 = self.cross_entropy_cost(activations=a, outputs=y)

        a = [np.array([0.3], float)]
        y = [np.array([0], float)]
        c2 = self.cross_entropy_cost(activations=a, outputs=y)
        self.assertGreaterEqual(c2, c1)

    def test_single_example_1_output_neuron_entropy_decreases(self):
        a = [np.array([0.1], float)]
        y = [np.array([0], float)]
        c1 = self.cross_entropy_cost(activations=a, outputs=y)

        a = [np.array([0.05], float)]
        y = [np.array([0], float)]
        c2 = self.cross_entropy_cost(activations=a, outputs=y)
        self.assertLessEqual(c2, c1)

    def test_single_example_many_output_neurons_no_entropy(self):
        a = [np.array([0, 0], float)]
        y = [np.array([0, 0], float)]
        c = self.cross_entropy_cost(activations=a, outputs=y)
        self.assertAlmostEqual(c, 0, places=4)

        a = [np.array([1, 1], float)]
        y = [np.array([1, 1], float)]
        c = self.cross_entropy_cost(activations=a, outputs=y)
        self.assertAlmostEqual(c, 0, places=4)

        a = [np.array([0.5, 0.5], float)]
        y = [np.array([0.5, 0.5], float)]
        c = self.cross_entropy_cost(activations=a, outputs=y)
        self.assertAlmostEqual(c, 0.6931 * 2, places=3)

    def test_single_example_many_output_neurons_max_entropy(self):
        from math import exp
        a = [np.array([1 - exp(-10), exp(-10)], float)]
        y = [np.array([0, 1], float)]
        c = self.cross_entropy_cost(activations=a, outputs=y)
        self.assertAlmostEqual(c, 20, places=4)

        a = [np.array([exp(-10), 1 - exp(-10)], float)]
        y = [np.array([1, 0], float)]
        c = self.cross_entropy_cost(activations=a, outputs=y)
        self.assertAlmostEqual(c, 20, places=4)

    def test_multiple_examples_multiple_output_neurons(self):
        a = [np.array([0, 0], float), np.array([0, 0], float)]
        y = [np.array([0, 0], float), np.array([0, 0], float)]
        c = self.cross_entropy_cost(activations=a, outputs=y)
        self.assertAlmostEqual(c, 0, places=4)

        a = [np.array([0.5, 0], float), np.array([0, 0.5], float), np.array([0, 0], float)]
        y = [np.array([0.5, 0], float), np.array([0, 0.5], float), np.array([0, 0], float)]
        c = self.cross_entropy_cost(activations=a, outputs=y)
        self.assertAlmostEqual(c, 0.6931 * 2 / 3, places=3)


class CrossEntropyGradientsTests(unittest.TestCase):
    def test_get_final_layer_error_for_1_element_vectors(self):
        cross_entropy = cost_functions.CrossEntropyCost()
        z_last = np.array([3], float)
        y = np.array([0], float)
        a_last = sigma(z_last)
        nabla = cross_entropy.get_final_layer_error(a_last, y, z_last, activation_function=Sigmoid)
        self.assertAlmostEqual(nabla[0], (a_last - y), places=2)

        z_last = np.array([-1], float)
        y = np.array([0.5], float)
        a_last = sigma(z_last)
        nabla = cross_entropy.get_final_layer_error(a_last, y, z_last, activation_function=Sigmoid)
        self.assertAlmostEqual(nabla[0], (a_last - y), places=2)

    def test_get_final_layer_error_for_arrays(self):
        cross_entropy = cost_functions.CrossEntropyCost()

        z_last = np.array([3, -1], float)
        y = np.array([0, 0.5], float)
        a_last = sigma(z_last)
        nabla = cross_entropy.get_final_layer_error(a_last, y, z_last, activation_function=Sigmoid)

        self.assertAlmostEqual(nabla[0], a_last[0] - y[0], places=2)
        self.assertAlmostEqual(nabla[1], a_last[1] - y[1],
                               places=2)
