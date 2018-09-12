import unittest
import numpy as np
from cost_functions import QuadraticCost, CrossEntropyCost, RegularizedCost
from main import NetFactory


class RegularizedCostTests(unittest.TestCase):
    def setUp(self):
        self.net = NetFactory.create_neural_net(sizes=[2, 3, 4])

    def check_case(self, cost_function, activations, outputs, reglambda, weights):
        regcost = RegularizedCost(neural_net=self.net,
                                  cost_function=cost_function,
                                  regularization_parameter=reglambda,
                                  weights=weights)

        c1 = cost_function.compute_cost(activations=activations, outputs=outputs)
        c2 = regcost.compute_cost(activations=activations, outputs=outputs)

        wsum = sum([(w ** 2).sum() for w in weights])
        n = len(outputs)
        self.assertAlmostEqual(c2, c1 + reglambda / 2.0 * wsum, places=3)

    def test_quadratic_regularized_on_single_example(self):
        activations = [np.array([-1, 1], float)]
        outputs = [np.array([0, 1], float)]

        reglambda = 1
        weights = [np.array([[-2]])]
        regcost = RegularizedCost(neural_net=self.net,
                                  cost_function=QuadraticCost(self.net),
                                  regularization_parameter=reglambda,
                                  weights=weights)

        c = regcost.compute_cost(activations=activations, outputs=outputs)

        self.assertAlmostEqual(c, 2.5, places=3)

    def test_regularized_quadratic_cost_for_2_examples(self):
        activations = [np.array([0.7, 0.6], float), np.array([1, 0], float)]
        outputs = [np.array([0.2, 0.5], float), np.array([0, 0], float)]

        weights = [np.array([[3, -1], [0.5, 2]], float),
                   np.array([[3, -1, 1]], float)]
        self.check_case(cost_function=QuadraticCost(self.net), activations=activations,
                        outputs=outputs, reglambda=1, weights=weights)

    def test_with_lambda_equal_to_zero(self):

        activations = [np.array([0.7, 0.6], float), np.array([1, 0], float)]
        outputs = [np.array([0.2, 0.5], float), np.array([0, 0], float)]

        weights = [np.array([[3, -1]])]
        self.check_case(cost_function=QuadraticCost(self.net), activations=activations,
                        outputs=outputs, reglambda=0, weights=weights)

    def test_regularized_xentropy(self):
        activations = [np.array([0.7, 0.6], float) ]
        outputs = [np.array([0.2, 0.5], float)]

        weights = [np.array([[3, -1]])]

        self.check_case(cost_function=CrossEntropyCost(self.net), activations=activations,
                        outputs=outputs, reglambda=1, weights=weights)

    def test_regularized_xentropy_with_many_examples(self):
        activations = [np.array([0.7, 0.6], float), np.array([0.2, 0])]
        outputs = [np.array([0.2, 0.5], float), np.array([0.5, 0.5])]

        weights = [np.array([[3, -1]])]

        self.check_case(cost_function=CrossEntropyCost(self.net), activations=activations,
                        outputs=outputs, reglambda=0.25, weights=weights)

    def test_get_lambda(self):
        weights = [np.array([[3, -1]])]

        regcost = RegularizedCost(neural_net=self.net,
                                  cost_function=QuadraticCost(self.net),
                                  regularization_parameter=2,
                                  weights=weights)

        self.assertEqual(regcost.get_lambda(), 2)
