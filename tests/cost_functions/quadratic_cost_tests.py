import unittest
import numpy as np
import cost_functions
from main import NeuralNet, NetFactory
from data_source import PreloadSource


class QuadraticCostTests(unittest.TestCase):
    def setUp(self):
        self.net = NetFactory.create_neural_net(sizes=[3, 1, 2])

    def test_quadratic_cost(self):
        activations = [np.array([0.7, 0.6], float), np.array([1, 0], float)]
        outputs = [np.array([0.2, 0.5], float), np.array([0, 0], float)]
        quadracost = cost_functions.QuadraticCost(neural_net=self.net)
        c = quadracost.compute_cost(activations=activations, outputs=outputs)
        self.assertAlmostEqual(c, 0.315, places=3)

    def test_quadratic_per_example(self):
        quadracost = cost_functions.QuadraticCost(neural_net=self.net)

        a = [np.array([0.5, 0.7], float)]
        y = [np.array([0.2, 0.1], float)]
        c = quadracost.compute_cost(activations=a, outputs=y)
        self.assertAlmostEqual(c, 0.225, places=3)

        a = [np.array([1, 0], float)]
        y = [np.array([0, 1], float)]
        c = quadracost.compute_cost(activations=a, outputs=y)
        self.assertAlmostEqual(c, 1, places=1)

    def test_get_cost_initial(self):
        nnet = NetFactory.create_neural_net(sizes=[1, 1, 1])

        xes = [np.array([-10], float), np.array([100], float)]
        ys = [np.array([0.5], float), np.array([0.75], float)]

        examples = PreloadSource((xes, ys))
        cost_func = cost_functions.QuadraticCost(nnet)
        cost = cost_func.get_cost(examples)
        self.assertAlmostEqual(cost, 1.0/64, places=4)

    def test_get_lambda(self):
        quadracost = cost_functions.QuadraticCost(neural_net=self.net)
        self.assertEqual(quadracost.get_lambda(), 0)
