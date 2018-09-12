import unittest
import numpy as np
from cost_functions import QuadraticCost, CrossEntropyCost
from main import NetFactory
from data_source import PreloadSource


class RegularizedCostTests(unittest.TestCase):
    def setUp(self):
        self.net = NetFactory.create_neural_net(sizes=[2, 3, 2])

    def check_case(self, cost_function, inputs, outputs, reglambda):
        src = PreloadSource((inputs, outputs))

        cost_func = cost_function(neural_net=self.net, l2_reg_term=0)
        c1 = cost_func.get_cost(data_src=src)
        reg_loss = cost_function(neural_net=self.net, l2_reg_term=reglambda)

        c2 = reg_loss.get_cost(data_src=src)

        wsum = sum([(w ** 2).sum() for w in self.net.weights()])

        self.assertAlmostEqual(c2, c1 + reglambda / 2.0 * wsum, places=3)

    def test_regularized_quadratic_cost_for_2_examples(self):
        inputs = [np.array([-1, 1], float), np.array([0, 0], float)]
        outputs = [np.array([1, 1], float), np.array([0, 0.5], float)]

        self.net.randomize_parameters()
        self.check_case(QuadraticCost, inputs, outputs, 5.5)

    def test_regularized_xentropy_with_many_examples(self):
        inputs = [np.array([0.7, 0.6], float),
                  np.array([0.2, 0], float)]
        outputs = [np.array([0.2, 0.5], float),
                   np.array([0.5, 0.5], float)]

        self.net.randomize_parameters()
        self.check_case(CrossEntropyCost, inputs, outputs, 5.5)

    def test_get_lambda(self):
        regcost = QuadraticCost(self.net)
        self.assertEqual(regcost.get_lambda(), 0)

        regcost = QuadraticCost(self.net, l2_reg_term=2)
        self.assertEqual(regcost.get_lambda(), 2)
