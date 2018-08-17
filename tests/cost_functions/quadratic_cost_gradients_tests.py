import unittest
import numpy as np
import cost_functions
from activation_functions import Sigmoid, Rectifier


class QuadraticCostGradientsTests(unittest.TestCase):
    def test_get_final_layer_error_for_1_element_vectors(self):
        quadratic = cost_functions.QuadraticCost()

        z_last = np.array([-1], float)
        z_last_prime = Rectifier.gradient(z_last)

        y = np.array([0.5], float)
        a_last = Rectifier.activation(z_last)
        nabla = quadratic.get_final_layer_error(a_last, y, z_last_prime)
        self.assertAlmostEqual(nabla[0], (a_last - y) * z_last_prime, places=2)

    def test_get_final_layer_error_for_arrays(self):
        quadratic = cost_functions.QuadraticCost()

        z_last = np.array([3, -1], float)
        z_last_prime = Sigmoid.gradient(z_last)
        y = np.array([0, 0.5], float)
        a_last = Sigmoid.activation(z_last)
        nabla = quadratic.get_final_layer_error(a_last, y, z_last_prime)

        self.assertAlmostEqual(nabla[0], (a_last[0] - y[0]) * z_last_prime[0], places=2)
        self.assertAlmostEqual(nabla[1], (a_last[1] - y[1]) * Sigmoid.gradient(z_last[1]),
                               places=2)

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
        z_prime = Rectifier.gradient(z)
        nabla = quadratic.get_error_in_layer(nabla_next, w_next, z_prime)

        expected_nabla = w_next.T.dot(nabla_next) * z_prime
        self.assertTrue(np.allclose(nabla, expected_nabla))
