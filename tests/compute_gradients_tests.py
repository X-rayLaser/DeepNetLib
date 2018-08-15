import unittest
import numpy as np
import helpers
import backprop_slow
from main import NeuralNet
import cost_functions
import activation_functions


class ComputeGradientsTests(unittest.TestCase):
    def compare_grads(self, grad1, grad2):
        self.assertTrue(helpers.gradients_equal(grad1, grad2))

    def test_compute_gradients_with_quadratic_cost(self):
        nnet = NeuralNet(layer_sizes=[4, 15, 10])
        nnet.randomize_parameters()
        examples = helpers.generate_random_examples(10, 4, 10)
        cost_function = nnet.get_cost_function()

        w_grad1, b_grad1 = cost_function.compute_gradients(examples=examples, neural_net=nnet)
        w_grad2, b_grad2 = backprop_slow.back_propagation_slow(examples=examples, neural_net=nnet)

        self.compare_grads(grad1=w_grad1, grad2=w_grad2)
        self.compare_grads(grad1=b_grad1, grad2=b_grad2)

    def test_compute_gradients_with_cross_entropy_cost(self):
        nnet = NeuralNet(layer_sizes=[4, 15, 10])
        nnet.randomize_parameters()
        nnet.set_cost_function(cost_function=cost_functions.CrossEntropyCost())
        examples = helpers.generate_random_examples(10, 4, 10)
        cost_function = nnet.get_cost_function()

        w_grad1, b_grad1 = cost_function.compute_gradients(examples=examples, neural_net=nnet)
        w_grad2, b_grad2 = backprop_slow.back_propagation_slow(examples=examples, neural_net=nnet)

        self.compare_grads(grad1=w_grad1, grad2=w_grad2)
        self.compare_grads(grad1=b_grad1, grad2=b_grad2)

    def test_with_rectifer_activation_and_quadratic_cost(self):
        nnet = NeuralNet(layer_sizes=[4, 15, 10])
        nnet.randomize_parameters()
        nnet.set_activation_function(activation=activation_functions.Rectifier)
        examples = helpers.generate_random_examples(10, 4, 10)
        cost_function = nnet.get_cost_function()

        w_grad1, b_grad1 = cost_function.compute_gradients(examples=examples, neural_net=nnet)
        w_grad2, b_grad2 = backprop_slow.back_propagation_slow(examples=examples, neural_net=nnet)

        self.compare_grads(grad1=w_grad1, grad2=w_grad2)
        self.compare_grads(grad1=b_grad1, grad2=b_grad2)

    def test_that_returned_type_is_array(self):
        nnet = NeuralNet(layer_sizes=[2, 1, 2])
        x = np.array([5, 2], float)
        y = np.array([0.25, 0], float)

        examples = ([x], [y])
        cost_function = nnet.get_cost_function()

        w_grad, b_grad = cost_function.compute_gradients(examples=examples, neural_net=nnet)
        self.assertIsInstance(w_grad, list)
        self.assertIsInstance(w_grad[0], np.ndarray)
        self.assertIsInstance(w_grad[1], np.ndarray)

        self.assertIsInstance(b_grad, list)
        self.assertIsInstance(b_grad[0], np.ndarray)
        self.assertIsInstance(b_grad[1], np.ndarray)

    def test_returns_correct_gradient_shape(self):
        nnet = NeuralNet(layer_sizes=[3, 2, 2, 5])
        x = np.array([5, 2, -0.5], float)
        y = np.array([0.25, 0, 0, 0.7, 0.2], float)
        examples = ([x], [y])
        w_grad, b_grad = backprop_slow.back_propagation_slow(examples=examples, neural_net=nnet)
        self.assertEqual(len(w_grad), 3)
        self.assertEqual(len(b_grad), 3)
        self.assertTupleEqual(w_grad[0].shape, (2, 3))
        self.assertTupleEqual(w_grad[1].shape, (2, 2))
        self.assertTupleEqual(w_grad[2].shape, (5, 2))

        self.assertTupleEqual(b_grad[0].shape, (2,))
        self.assertTupleEqual(b_grad[1].shape, (2,))
        self.assertTupleEqual(b_grad[2].shape, (5,))

    def test_with_regularization(self):
        nnet = NeuralNet(layer_sizes=[4, 15, 10])
        nnet.randomize_parameters()
        nnet.set_regularization(reg_lambda=2)
        examples = helpers.generate_random_examples(10, 4, 10)
        cost_function = nnet.get_cost_function()

        w_grad1, b_grad1 = cost_function.compute_gradients(examples=examples, neural_net=nnet)
        w_grad2, b_grad2 = backprop_slow.back_propagation_slow(examples=examples, neural_net=nnet)

        self.compare_grads(grad1=w_grad1, grad2=w_grad2)
        self.compare_grads(grad1=b_grad1, grad2=b_grad2)
