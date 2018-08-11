import unittest
import numpy as np
from activation_functions import sigma, sigma_prime, Rectifier, Sigmoid


class SigmoidTests(unittest.TestCase):
    def test_sigma(self):
        self.assertAlmostEqual(Sigmoid.activation(0), 0.5, places=2)
        self.assertAlmostEqual(Sigmoid.activation(50), 1, places=2)
        self.assertAlmostEqual(Sigmoid.activation(-50), 0, places=2)

        self.assertAlmostEqual(Sigmoid.activation(1), 0.731, places=2)
        self.assertAlmostEqual(Sigmoid.activation(-1), 0.2689, places=2)

    def test_sigma_prime(self):
        self.assertAlmostEqual(Sigmoid.gradient(0), 0.25, places=3)
        self.assertAlmostEqual(Sigmoid.gradient(-50), 0, places=3)
        self.assertAlmostEqual(Sigmoid.gradient(50), 0, places=3)

        self.assertAlmostEqual(Sigmoid.gradient(50), Sigmoid.activation(50) * (1 - Sigmoid.activation(50)), places=3)


class RectifierActivationTests(unittest.TestCase):
    def test_for_negative_input_values(self):
        a = Rectifier.activation(np.array([-0.5]))
        self.assertEqual(a.tolist(), [0])

        a = Rectifier.activation(np.array([-10.5]))
        self.assertEqual(a.tolist(), [0])

    def test_for_positive_input_values(self):
        a = Rectifier.activation(np.array([0.001]))
        self.assertEqual(a.tolist(), [0.001])

        a = Rectifier.activation(np.array([2]))
        self.assertEqual(a.tolist(), [2])

        a = Rectifier.activation(np.array([10*3]))
        self.assertEqual(a.tolist(), [10*3])

    def test_for_zero_input_values(self):
        a = Rectifier.activation(np.array([0]))
        self.assertEqual(a.tolist(), [0])

    def test_for_vectors(self):
        mylist = [-10, 0, 2, 10**20]
        a = Rectifier.activation(np.array(mylist, float))
        self.assertEqual(a.tolist(), [0, 0, 2, 10**20])


class RectifierGradientTests(unittest.TestCase):
    def test_for_non_positive_values(self):
        a = Rectifier.gradient(np.array([-0.5]))
        self.assertEqual(a.tolist(), [0])

        a = Rectifier.gradient(np.array([0]))
        self.assertEqual(a.tolist(), [0])

        a = Rectifier.gradient(np.array([-10.5]))
        self.assertEqual(a.tolist(), [0])

    def test_for_positive_input_values(self):
        a = Rectifier.gradient(np.array([0.001]))
        self.assertEqual(a.tolist(), [1])

        a = Rectifier.gradient(np.array([30]))
        self.assertEqual(a.tolist(), [1])

        a = Rectifier.gradient(np.array([10*3]))
        self.assertEqual(a.tolist(), [1])

    def test_for_vectors(self):
        mylist = [-10, 0, 2, 10**20]
        a = Rectifier.gradient(np.array(mylist, float))
        self.assertEqual(a.tolist(), [0, 0, 1, 1])
