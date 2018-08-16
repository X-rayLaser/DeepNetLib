import unittest
import numpy as np
import cost_functions
from activation_functions import sigma, Softmax, Sigmoid, Rectifier


class CrossEntropyGradientsTests(unittest.TestCase):
    def test_get_final_layer_error_for_1_element_vectors(self):
        cross_entropy = cost_functions.CrossEntropyCost()
        z_last = np.array([3], float)
        z_last_prime = Sigmoid.gradient(z_last)

        y = np.array([0], float)
        a_last = sigma(z_last)
        nabla = cross_entropy.get_final_layer_error(a_last, y, z_last_prime)
        self.assertAlmostEqual(nabla[0], (a_last - y), places=2)

        z_last = np.array([-1], float)
        z_last_prime = Rectifier.gradient(z_last)
        y = np.array([0.5], float)
        a_last = sigma(z_last)
        nabla = cross_entropy.get_final_layer_error(a_last, y, z_last_prime)
        self.assertAlmostEqual(nabla[0], (a_last - y), places=2)

    def test_get_final_layer_error_for_arrays(self):
        cross_entropy = cost_functions.CrossEntropyCost()

        z_last = np.array([3, -1], float)
        z_last_prime = Softmax.gradient(z_last)

        y = np.array([0, 0.5], float)
        a_last = sigma(z_last)
        nabla = cross_entropy.get_final_layer_error(a_last, y, z_last_prime)

        self.assertAlmostEqual(nabla[0], a_last[0] - y[0], places=2)
        self.assertAlmostEqual(nabla[1], a_last[1] - y[1],
                               places=2)
