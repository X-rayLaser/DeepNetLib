import unittest
import numpy as np
import cost_functions
from neural_net import NetFactory


class CrossEntropyCostTests(unittest.TestCase):
    def cross_entropy_cost(self, a, y):
        net = NetFactory.create_neural_net(sizes=[1, 2, 3])
        cost_func = cost_functions.CrossEntropyCost(neural_net=net)
        return cost_func.individual_cost(a[0], y[0])

    def test_single_example_1_output_neuron_no_entropy(self):
        a = [np.array([0], float)]
        y = [np.array([0], float)]
        c = self.cross_entropy_cost(a, y)
        self.assertAlmostEqual(c, 0, places=4)

        a = [np.array([1], float)]
        y = [np.array([1], float)]
        c = self.cross_entropy_cost(a, y)
        self.assertAlmostEqual(c, 0, places=4)

        a = [np.array([0.5], float)]
        y = [np.array([0.5], float)]
        c = self.cross_entropy_cost(a, y)
        self.assertAlmostEqual(c, 0.6931, places=4)

    def test_single_example_1_output_neuron_max_entropy(self):
        from math import exp
        a = [np.array([1 - exp(-10)], float)]
        y = [np.array([0], float)]
        c = self.cross_entropy_cost(a, y)
        self.assertAlmostEqual(c, 10, places=4)

        a = [np.array([exp(-10)], float)]
        y = [np.array([1], float)]
        c = self.cross_entropy_cost(a, y)
        self.assertAlmostEqual(c, 10, places=4)

    def test_single_example_1_output_neuron_entropy_increases(self):
        a = [np.array([0.1], float)]
        y = [np.array([0], float)]
        c1 = self.cross_entropy_cost(a, y)

        a = [np.array([0.3], float)]
        y = [np.array([0], float)]
        c2 = self.cross_entropy_cost(a, y)
        self.assertGreaterEqual(c2, c1)

    def test_single_example_1_output_neuron_entropy_decreases(self):
        a = [np.array([0.1], float)]
        y = [np.array([0], float)]
        c1 = self.cross_entropy_cost(a, y)

        a = [np.array([0.05], float)]
        y = [np.array([0], float)]
        c2 = self.cross_entropy_cost(a, y)
        self.assertLessEqual(c2, c1)

    def test_single_example_many_output_neurons_no_entropy(self):
        a = [np.array([0, 0], float)]
        y = [np.array([0, 0], float)]
        c = self.cross_entropy_cost(a, y)
        self.assertAlmostEqual(c, 0, places=4)

        a = [np.array([1, 1], float)]
        y = [np.array([1, 1], float)]
        c = self.cross_entropy_cost(a, y)
        self.assertAlmostEqual(c, 0, places=4)

        a = [np.array([0.5, 0.5], float)]
        y = [np.array([0.5, 0.5], float)]
        c = self.cross_entropy_cost(a, y)
        self.assertAlmostEqual(c, 0.6931 * 2, places=3)

    def test_single_example_many_output_neurons_max_entropy(self):
        from math import exp
        a = [np.array([1 - exp(-10), exp(-10)], float)]
        y = [np.array([0, 1], float)]
        c = self.cross_entropy_cost(a, y)
        self.assertAlmostEqual(c, 20, places=4)

        a = [np.array([exp(-10), 1 - exp(-10)], float)]
        y = [np.array([1, 0], float)]
        c = self.cross_entropy_cost(a, y)
        self.assertAlmostEqual(c, 20, places=4)

    def test_get_lambda(self):
        net = NetFactory.create_neural_net(sizes=[1, 2, 2])
        cost_func = cost_functions.CrossEntropyCost(neural_net=net)
        self.assertEqual(cost_func.get_lambda(), 0)
