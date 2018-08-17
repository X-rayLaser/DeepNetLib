import unittest
import numpy as np
from main import NeuralNet
from gradient_descent import GradientDescent, StochasticGradientDescent
from backprop import BackPropagation
from gradient_calculator import BackPropagationBasedCalculator


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
        calculator = BackPropagationBasedCalculator(examples=self.examples,
                                                    neural_net=self.nnet)

        for i in range(5):
            w_grad, b_grad = calculator.compute_gradients()
            self.grad_descent.update_weights(weight_gradient=w_grad)
            self.grad_descent.update_biases(bias_gradient=b_grad)

        cost_after = self.nnet.get_cost(self.examples)
        self.assertLess(cost_after, cost_before)

    def test_update_biases_decreases_cost(self):
        cost_before = self.nnet.get_cost(self.examples)
        calculator = BackPropagationBasedCalculator(examples=self.examples,
                                                    neural_net=self.nnet)

        for i in range(5):
            w_grad, b_grad = calculator.compute_gradients()
            self.grad_descent.update_biases(bias_gradient=b_grad)

        cost_after = self.nnet.get_cost(self.examples)
        self.assertLess(cost_after, cost_before)

    def test_update_with_multiple_examples(self):
        self.nnet.randomize_parameters()

        self.examples = ([np.array([5, 2], float), np.array([5, 22], float)],
                         [np.array([0.25, 0, 1], float), np.array([0.5, 1, 0], float)])

        cost_before = self.nnet.get_cost(self.examples)
        calculator = BackPropagationBasedCalculator(examples=self.examples,
                                                    neural_net=self.nnet)
        for i in range(10):
            w_grad, b_grad = calculator.compute_gradients()
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


class GradientLearningRateTests(unittest.TestCase):
    def setUp(self):
        x = np.array([5, 2], float)
        y = np.array([0.25, 0, 1], float)
        self.examples = ([x], [y])
        nnet = NeuralNet(layer_sizes=[2, 3, 3])
        self.nnet = nnet
        self.Descent = GradientDescent

    def test_learning_rate_is_set(self):
        gd = self.Descent(neural_net=self.nnet, learning_rate=2)
        self.assertEqual(gd._rate, 2)
        gd = self.Descent(neural_net=self.nnet, learning_rate=0.1)
        self.assertEqual(gd._rate, 0.1)

    def test_forbid_weird_learning_rates(self):
        self.assertRaises(
            GradientDescent.InvalidLearningRate,
            lambda: self.Descent(neural_net=self.nnet, learning_rate=0)
        )

        self.assertRaises(
            GradientDescent.InvalidLearningRate,
            lambda: self.Descent(neural_net=self.nnet, learning_rate=-20)
        )


class StochasticGradientLearningRateTests(GradientLearningRateTests):
    def setUp(self):
        GradientLearningRateTests.setUp(self)
        self.Descent = StochasticGradientDescent


class StochasticGradientDescentTests(GradientDescentTest):
    def setUp(self):
        x = np.array([5, 2], float)
        y = np.array([0.25, 0, 1], float)
        self.examples = ([x], [y])
        nnet = NeuralNet(layer_sizes=[2, 3, 3])
        nnet.randomize_parameters()
        self.nnet = nnet
        self.grad_descent = StochasticGradientDescent(neural_net=nnet)

    def test_mini_batch_is_set(self):
        gd = StochasticGradientDescent(neural_net=self.nnet, batch_size=20)
        self.assertEqual(gd._batch_size, 20)
        gd = StochasticGradientDescent(neural_net=self.nnet, batch_size=1)
        self.assertEqual(gd._batch_size, 1)

    def test_with_wrong_batch_size(self):
        self.assertRaises(
            StochasticGradientDescent.InvalidBatchSize,
            lambda: StochasticGradientDescent(neural_net=self.nnet, batch_size=0)
        )
        self.assertRaises(
            StochasticGradientDescent.InvalidBatchSize,
            lambda: StochasticGradientDescent(neural_net=self.nnet, batch_size=-1)
        )
        self.assertRaises(
            StochasticGradientDescent.InvalidBatchSize,
            lambda: StochasticGradientDescent(neural_net=self.nnet, batch_size=-10)
        )

        self.assertRaises(
            StochasticGradientDescent.InvalidBatchSize,
            lambda: StochasticGradientDescent(neural_net=self.nnet, batch_size=3.5)
        )
