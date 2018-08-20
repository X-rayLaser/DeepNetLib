import unittest
from main import NeuralNet, NetFactory
import cost_functions


class GetCostFunction(unittest.TestCase):
    def test_with_default_cost(self):
        nnet = NetFactory.create_neural_net(sizes=[2, 1, 2])
        cost_func = nnet.get_cost_function()
        self.assertIsInstance(cost_func, cost_functions.QuadraticCost)

    def test_after_setting_other_cost(self):
        nnet = NetFactory.create_neural_net(sizes=[2, 1, 2])
        nnet.set_cost_function(cost_functions.QuadraticCost())
        self.assertIsInstance(nnet.get_cost_function(), cost_functions.QuadraticCost)

        nnet.set_cost_function(cost_functions.CrossEntropyCost())
        self.assertIsInstance(nnet.get_cost_function(), cost_functions.CrossEntropyCost)
