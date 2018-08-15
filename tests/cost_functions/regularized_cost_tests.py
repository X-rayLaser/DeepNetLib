import unittest
import numpy as np
from cost_functions import QuadraticCost, RegularizedCost
from main import NeuralNet


class RegularizedCostTests(unittest.TestCase):
    def test_quadratic_regularized_on_single_example(self):
        activations = [np.array([-1, 1], float)]
        outputs = [np.array([0, 1], float)]

        reglambda = 1
        weights_flatten = np.array([-2])
        regcost = RegularizedCost(cost_function=QuadraticCost(),
                                  regularization_parameter=reglambda,
                                  weights=weights_flatten)

        c = regcost.compute_cost(activations=activations, outputs=outputs)

        self.assertAlmostEqual(c, 2.5, places=3)

    def test_regularized_quadratic_cost_for_2_examples(self):
        activations = [np.array([0.7, 0.6], float), np.array([1, 0], float)]
        outputs = [np.array([0.2, 0.5], float), np.array([0, 0], float)]
        quadcost = QuadraticCost()

        reglambda = 1
        weights_flatten = np.array([3, -1, 0.5])
        regcost = RegularizedCost(cost_function=QuadraticCost(),
                                  regularization_parameter=reglambda,
                                  weights=weights_flatten)

        c1 = quadcost.compute_cost(activations=activations, outputs=outputs)
        c2 = regcost.compute_cost(activations=activations, outputs=outputs)

        wsum = (weights_flatten ** 2).sum()
        self.assertAlmostEqual(c2, c1 + reglambda / (2 * 2) * wsum, places=3)

    def test_with_lambda_equal_to_zero(self):
        activations = [np.array([0.7, 0.6], float), np.array([1, 0], float)]
        outputs = [np.array([0.2, 0.5], float), np.array([0, 0], float)]
        quadcost = QuadraticCost()

        reglambda = 0
        weights_flatten = np.array([3, -1, 0.5])
        regcost = RegularizedCost(cost_function=QuadraticCost(),
                                  regularization_parameter=reglambda,
                                  weights=weights_flatten)

        c1 = quadcost.compute_cost(activations=activations, outputs=outputs)
        c2 = regcost.compute_cost(activations=activations, outputs=outputs)

        self.assertAlmostEqual(c2, c1, places=3)
