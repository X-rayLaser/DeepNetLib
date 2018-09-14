import unittest
import numpy as np
import cost_functions
from activation_functions import Sigmoid, Rectifier
from neural_net import NetFactory


class QuadraticCostGradientsTests(unittest.TestCase):
    def setUp(self):
        self.net = NetFactory.create_neural_net(sizes=[2, 1, 1])

    def test_get_final_layer_error_for_1_element_vectors(self):
        quadratic = cost_functions.QuadraticCost(neural_net=self.net)

        z_last = np.array([-1], float)
        z_last_prime = Rectifier.gradient(z_last)

        y = np.array([0.5], float)
        a_last = Rectifier.activation(z_last)
        nabla = quadratic.get_final_layer_error(a_last, y, z_last_prime)
        self.assertAlmostEqual(nabla[0], (a_last - y) * z_last_prime, places=2)

    def test_get_final_layer_error_for_arrays(self):
        quadratic = cost_functions.QuadraticCost(neural_net=self.net)

        z_last = np.array([3, -1], float)
        z_last_prime = Sigmoid.gradient(z_last)
        y = np.array([0, 0.5], float)
        a_last = Sigmoid.activation(z_last)
        nabla = quadratic.get_final_layer_error(a_last, y, z_last_prime)

        self.assertAlmostEqual(nabla[0], (a_last[0] - y[0]) * z_last_prime[0], places=2)
        self.assertAlmostEqual(nabla[1], (a_last[1] - y[1]) * Sigmoid.gradient(z_last[1]),
                               places=2)
