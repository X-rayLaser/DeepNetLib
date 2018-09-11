import unittest
import numpy as np
import helpers
from main import NetFactory
from gradient_calculator import NumericalCalculator
from cost_functions import QuadraticCost
from data_source import PreloadSource


class BackpropSlowTests(unittest.TestCase):
    def compare_grads(self, grad1, grad2):
        self.assertTrue(helpers.gradients_equal(grad1, grad2))

    def test_back_propagation_slow(self):
        nnet = NetFactory.create_neural_net(sizes=[1, 1, 1])
        cost_func = QuadraticCost(neural_net=nnet)
        x = np.array([5], float)
        y = np.array([0.25], float)
        examples = ([x], [y])

        numerical = NumericalCalculator(data_src=PreloadSource(examples),
                                        neural_net=nnet,
                                        cost_function=cost_func)
        w_grad, b_grad = numerical.compute_gradients()

        w_grad_expected = [np.array([[0]], float), np.array([[1/32]], float)]
        b_grad_expected = [np.array([[0]], float), np.array([[1/16]], float)]

        self.compare_grads(w_grad, w_grad_expected)
        self.compare_grads(b_grad, b_grad_expected)

    def test_back_propagation_slow_type_array(self):
        nnet = NetFactory.create_neural_net(sizes=[2, 1, 2])
        cost_func = QuadraticCost(neural_net=nnet)

        x = np.array([5, 2], float)
        y = np.array([0.25, 0], float)

        examples = ([x], [y])
        numerical = NumericalCalculator(data_src=PreloadSource(examples),
                                        neural_net=nnet,
                                        cost_function=cost_func)

        w_grad, b_grad = numerical.compute_gradients()
        self.assertIsInstance(w_grad, list)
        self.assertIsInstance(w_grad[0], np.ndarray)
        self.assertIsInstance(w_grad[1], np.ndarray)

        self.assertIsInstance(b_grad, list)
        self.assertIsInstance(b_grad[0], np.ndarray)
        self.assertIsInstance(b_grad[1], np.ndarray)

    def test_back_propagation_slow_shape(self):
        nnet = NetFactory.create_neural_net(sizes=[3, 2, 2, 5])
        cost_func = QuadraticCost(neural_net=nnet)

        x = np.array([5, 2, -0.5], float)
        y = np.array([0.25, 0, 0, 0.7, 0.2], float)
        examples = ([x], [y])
        numerical = NumericalCalculator(data_src=PreloadSource(examples),
                                        neural_net=nnet,
                                        cost_function=cost_func)
        w_grad, b_grad = numerical.compute_gradients()

        self.assertEqual(len(w_grad), 3)
        self.assertEqual(len(b_grad), 3)
        self.assertTupleEqual(w_grad[0].shape, (2, 3))
        self.assertTupleEqual(w_grad[1].shape, (2, 2))
        self.assertTupleEqual(w_grad[2].shape, (5, 2))

        self.assertTupleEqual(b_grad[0].shape, (2,))
        self.assertTupleEqual(b_grad[1].shape, (2,))
        self.assertTupleEqual(b_grad[2].shape, (5,))
