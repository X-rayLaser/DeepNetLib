import unittest
import numpy as np
from main import NeuralNet
from gradient_descent import GradientDescent, StochasticGradientDescent
from backprop import back_propagation


class GradientDescentTest(unittest.TestCase):
    def setUp(self):
        x = np.array([5, 2], float)
        y = np.array([0.25, 0, 1], float)
        self.examples = ([x], [y])
        nnet = NeuralNet(layer_sizes=[2, 3, 3])
        nnet.randomize_parameters()
        self.nnet = nnet
        self.grad_descent = GradientDescent(neural_net=nnet)

    def test_init(self):
        nnet = NeuralNet(layer_sizes=[2, 3, 5])
        grad_descent = GradientDescent(neural_net=nnet)

    def test_update_weights_decreases_cost(self):
        cost_before = self.nnet.get_cost(self.examples)

        for i in range(5):
            w_grad, b_grad = back_propagation(examples=self.examples, neural_net=self.nnet)
            self.grad_descent.update_weights(weight_gradient=w_grad)
            self.grad_descent.update_biases(bias_gradient=b_grad)

        cost_after = self.nnet.get_cost(self.examples)
        self.assertLess(cost_after, cost_before)

    def test_update_biases_decreases_cost(self):
        cost_before = self.nnet.get_cost(self.examples)

        for i in range(5):
            w_grad, b_grad = back_propagation(examples=self.examples, neural_net=self.nnet)
            self.grad_descent.update_biases(bias_gradient=b_grad)

        cost_after = self.nnet.get_cost(self.examples)
        self.assertLess(cost_after, cost_before)

    def test_update_with_multiple_examples(self):
        self.nnet.randomize_parameters()

        self.examples = ([np.array([5, 2], float), np.array([5, 22], float)],
                         [np.array([0.25, 0, 1], float), np.array([0.5, 1, 0], float)])

        cost_before = self.nnet.get_cost(self.examples)

        for i in range(10):
            w_grad, b_grad = back_propagation(examples=self.examples, neural_net=self.nnet)
            self.grad_descent.update_weights(weight_gradient=w_grad)
            self.grad_descent.update_biases(bias_gradient=b_grad)

        cost_after = self.nnet.get_cost(self.examples)
        self.assertLess(cost_after, cost_before)

    def test_training_epoch_1_example(self):
        cost_before = self.nnet.get_cost(self.examples)
        self.grad_descent.training_epoch(examples=self.examples)
        cost_after = self.nnet.get_cost(self.examples)
        self.assertLess(cost_after, cost_before)

    def test_training_epoch_2_examples(self):
        self.examples = ([np.array([5, 2], float), np.array([5, 22], float)],
                         [np.array([0.25, 0, 1], float), np.array([0.5, 1, 0], float)])
        cost_before = self.nnet.get_cost(self.examples)
        self.grad_descent.training_epoch(examples=self.examples)
        cost_after = self.nnet.get_cost(self.examples)
        self.assertLess(cost_after, cost_before)

    def test_training_with_initially_random_parameters(self):
        self.nnet.randomize_parameters()
        cost_before = self.nnet.get_cost(self.examples)
        self.grad_descent.training_epoch(examples=self.examples)
        cost_after = self.nnet.get_cost(self.examples)
        self.assertLess(cost_after, cost_before)

    def test_train_for_few_epoch(self):
        cost_before = self.nnet.get_cost(self.examples)
        self.grad_descent.train(examples=self.examples, nepochs=2)
        cost_after = self.nnet.get_cost(self.examples)
        self.assertLess(cost_after, cost_before)


class StochasticGradientDescentTests(GradientDescentTest):
    def setUp(self):
        x = np.array([5, 2], float)
        y = np.array([0.25, 0, 1], float)
        self.examples = ([x], [y])
        nnet = NeuralNet(layer_sizes=[2, 3, 3])
        nnet.randomize_parameters()
        self.nnet = nnet
        self.grad_descent = StochasticGradientDescent(neural_net=nnet)
