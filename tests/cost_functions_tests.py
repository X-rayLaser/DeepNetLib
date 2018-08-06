import unittest
import numpy as np
import cost_functions
from main import NeuralNet


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
