import unittest
import numpy as np
from activation_functions import Rectifier, Sigmoid, Softmax


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


class SoftmaxTests(unittest.TestCase):
    def test_returns_array_of_valid_shape(self):
        z = np.array([1, 2], float)
        a = Softmax.activation(z)
        self.assertTupleEqual(a.shape, z.shape)

    def test_activations_in_correct_range(self):
        z = np.array([-1000, 0.1, 2, 200], float)
        a = Softmax.activation(z)

        self.assertTrue(np.all(0 <= a) and np.all(a <= 1))

    def test_results_add_to_1(self):
        z = np.array([-3, 0.1, 1, 20], float)
        a = Softmax.activation(z)
        self.assertAlmostEqual(a.sum(), 1)

    def test_for_2_element_vectors(self):
        z = np.array([1, 2], float)
        a = Softmax.activation(z)
        self.assertTrue(
            np.allclose(a, np.array([0.268941, 0.731058], float), )
        )

        z = np.array([0, 2], float)
        a = Softmax.activation(z)
        self.assertTrue(
            np.allclose(a, np.array([0.1192029, 0.880797], float), )
        )

    def test_on_vectors_with_huge_components(self):
        z = np.array([np.finfo(float).max, 2, np.finfo(float).max / 2], float)
        # won't raise OverflowError
        a = Softmax.activation(z)


class SoftmaxGradientTests(unittest.TestCase):
    def test_returns_jacobian_matrix_of_valid_shape(self):
        z = np.array([1, 2, -2], float)
        j = Softmax.gradient(z)
        self.assertTupleEqual(j.shape, (3, 3))

        z = np.array([1, 2], float)
        j = Softmax.gradient(z)
        self.assertTupleEqual(j.shape, (2, 2))

    def test_derivatives_with_equal_indices_in_jacobian_matrix(self):
        z = np.array([1, -1.5], float)
        j = Softmax.gradient(z)

        s = Softmax.activation(z)
        self.assertEqual(j[0, 0], s[0] * (1 - s[0]))

        s = Softmax.activation(z)
        self.assertEqual(j[1, 1], s[1] * (1 - s[1]))

    def test_derivatives_with_different_indices_in_jacobian_matrix(self):
        z = np.array([1, -1.5], float)
        j = Softmax.gradient(z)

        s = Softmax.activation(z)
        self.assertEqual(j[0, 1], s[0] * s[1])

        s = Softmax.activation(z)
        self.assertEqual(j[1, 0], s[1] * s[0])
