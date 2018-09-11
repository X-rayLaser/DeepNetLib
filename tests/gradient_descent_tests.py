import unittest
import numpy as np
from main import NetFactory
from gradient_descent import GradientDescent, StochasticGradientDescent
from gradient_calculator import BackPropagationBasedCalculator
from cost_functions import QuadraticCost
import helpers
from data_source import PreloadSource


class GradientDescentTest(unittest.TestCase):
    def setUp(self):
        x = np.array([5, 2], float)
        y = np.array([0.25, 0, 1], float)
        self.examples = PreloadSource(([x], [y]))
        nnet = NetFactory.create_neural_net(sizes=[2, 3, 3])
        nnet.randomize_parameters()
        self.nnet = nnet
        cost_func = QuadraticCost(nnet)
        self.grad_descent = GradientDescent(neural_net=nnet,
                                            cost_function=cost_func)

    def test_init(self):
        nnet = NetFactory.create_neural_net(sizes=[2, 3, 5])
        cost_func = QuadraticCost(nnet)
        grad_descent = GradientDescent(neural_net=nnet,
                                       cost_function=cost_func)

    def test_update_weights_decreases_cost(self):
        cost_func = QuadraticCost(self.nnet)
        cost_before = cost_func.get_cost(self.examples)
        calculator = BackPropagationBasedCalculator(data_src=self.examples,
                                                    cost_function=cost_func,
                                                    neural_net=self.nnet)

        for i in range(5):
            w_grad, b_grad = calculator.compute_gradients()
            self.grad_descent.update_weights(weight_gradient=w_grad)
            self.grad_descent.update_biases(bias_gradient=b_grad)

        cost_after = cost_func.get_cost(self.examples)
        self.assertLess(cost_after, cost_before)

    def test_update_biases_decreases_cost(self):
        cost_func = QuadraticCost(self.nnet)
        cost_before = cost_func.get_cost(self.examples)
        calculator = BackPropagationBasedCalculator(data_src=self.examples,
                                                    cost_function=cost_func,
                                                    neural_net=self.nnet)

        for i in range(5):
            w_grad, b_grad = calculator.compute_gradients()
            self.grad_descent.update_biases(bias_gradient=b_grad)

        cost_after = cost_func.get_cost(self.examples)
        self.assertLess(cost_after, cost_before)

    def test_update_with_multiple_examples(self):
        self.nnet.randomize_parameters()

        self.examples = PreloadSource((
            [np.array([5, 2], float), np.array([5, 22], float)],
            [np.array([0.25, 0, 1], float), np.array([0.5, 1, 0], float)])
        )

        cost_func = QuadraticCost(self.nnet)
        cost_before = cost_func.get_cost(self.examples)
        calculator = BackPropagationBasedCalculator(data_src=self.examples,
                                                    cost_function=cost_func,
                                                    neural_net=self.nnet)
        for i in range(10):
            w_grad, b_grad = calculator.compute_gradients()
            self.grad_descent.update_weights(weight_gradient=w_grad)
            self.grad_descent.update_biases(bias_gradient=b_grad)

        cost_after = cost_func.get_cost(self.examples)
        self.assertLess(cost_after, cost_before)

    def test_training_epoch_1_example(self):
        cost_func = QuadraticCost(self.nnet)
        cost_before = cost_func.get_cost(self.examples)
        self.grad_descent.training_epoch(data_src=self.examples)
        cost_after = cost_func.get_cost(self.examples)
        self.assertLess(cost_after, cost_before)

    def test_training_epoch_2_examples(self):
        cost_func = QuadraticCost(self.nnet)
        self.examples = PreloadSource((
            [np.array([5, 2], float), np.array([5, 22], float)],
            [np.array([0.25, 0, 1], float), np.array([0.5, 1, 0], float)])
        )
        cost_before = cost_func.get_cost(self.examples)
        self.grad_descent.training_epoch(data_src=self.examples)
        cost_after = cost_func.get_cost(self.examples)
        self.assertLess(cost_after, cost_before)

    def test_training_with_initially_random_parameters(self):
        cost_func = QuadraticCost(self.nnet)
        self.nnet.randomize_parameters()
        cost_before = cost_func.get_cost(self.examples)
        self.grad_descent.training_epoch(data_src=self.examples)
        cost_after = cost_func.get_cost(self.examples)
        self.assertLess(cost_after, cost_before)

    def test_train_for_few_epoch(self):
        cost_func = QuadraticCost(self.nnet)
        cost_before = cost_func.get_cost(self.examples)
        self.grad_descent.train(data_src=self.examples, nepochs=2)
        cost_after = cost_func.get_cost(self.examples)
        self.assertLess(cost_after, cost_before)


class GradientLearningRateTests(unittest.TestCase):
    def setUp(self):
        x = np.array([5, 2], float)
        y = np.array([0.25, 0, 1], float)
        self.examples = PreloadSource(([x], [y]))
        nnet = NetFactory.create_neural_net(sizes=[2, 3, 3])
        self.nnet = nnet
        self.Descent = GradientDescent
        self.cost_function = QuadraticCost(self.nnet)

    def test_learning_rate_is_set(self):
        cost_function = self.cost_function
        gd = self.Descent(neural_net=self.nnet, cost_function=cost_function, learning_rate=2)
        self.assertEqual(gd._rate, 2)
        gd = self.Descent(neural_net=self.nnet, cost_function=cost_function, learning_rate=0.1)
        self.assertEqual(gd._rate, 0.1)

    def test_forbid_weird_learning_rates(self):
        cost_function = self.cost_function

        self.assertRaises(
            GradientDescent.InvalidLearningRate,
            lambda: self.Descent(neural_net=self.nnet,
                                 cost_function=cost_function,
                                 learning_rate=0)
        )

        self.assertRaises(
            GradientDescent.InvalidLearningRate,
            lambda: self.Descent(neural_net=self.nnet,
                                 cost_function=cost_function,
                                 learning_rate=-20)
        )


class StochasticGradientLearningRateTests(GradientLearningRateTests):
    def setUp(self):
        GradientLearningRateTests.setUp(self)
        self.Descent = StochasticGradientDescent


class StochasticGradientDescentTests(GradientDescentTest):
    def setUp(self):
        x = np.array([5, 2], float)
        y = np.array([0.25, 0, 1], float)
        self.examples = PreloadSource(([x], [y]))
        nnet = NetFactory.create_neural_net(sizes=[2, 3, 3])
        nnet.randomize_parameters()
        self.nnet = nnet

        cost_function = QuadraticCost(self.nnet)
        self.grad_descent = StochasticGradientDescent(neural_net=nnet,
                                                      cost_function=cost_function)

    def test_mini_batch_is_set(self):
        cost_function = QuadraticCost(self.nnet)

        gd = StochasticGradientDescent(neural_net=self.nnet,
                                       cost_function=cost_function,
                                       batch_size=20)
        self.assertEqual(gd._batch_size, 20)
        gd = StochasticGradientDescent(neural_net=self.nnet,
                                       cost_function=cost_function,
                                       batch_size=1)
        self.assertEqual(gd._batch_size, 1)

    def test_with_wrong_batch_size(self):
        cost_function = QuadraticCost(self.nnet)

        self.assertRaises(
            StochasticGradientDescent.InvalidBatchSize,
            lambda: StochasticGradientDescent(neural_net=self.nnet,
                                              cost_function=cost_function,
                                              batch_size=0)
        )
        self.assertRaises(
            StochasticGradientDescent.InvalidBatchSize,
            lambda: StochasticGradientDescent(neural_net=self.nnet,
                                              cost_function=cost_function,
                                              batch_size=-1)
        )
        self.assertRaises(
            StochasticGradientDescent.InvalidBatchSize,
            lambda: StochasticGradientDescent(neural_net=self.nnet,
                                              cost_function=cost_function,
                                              batch_size=-10)
        )

        self.assertRaises(
            StochasticGradientDescent.InvalidBatchSize,
            lambda: StochasticGradientDescent(neural_net=self.nnet,
                                              cost_function=cost_function, batch_size=3.5)
        )


class NeuralNetTrain(unittest.TestCase):
    def test_gives_correct_output_on_training_data(self):
        nnet = NetFactory.create_neural_net(sizes=[1, 1, 1])
        cost_func = QuadraticCost(neural_net=nnet)
        gd = GradientDescent(neural_net=nnet, cost_function=cost_func)

        xes = [np.array([-10], float), np.array([100], float)]
        ys = [np.array([0.5], float), np.array([0.75], float)]

        gd.train(data_src=PreloadSource((xes, ys)), nepochs=100)

        for i in range(len(xes)):
            res = nnet.feed(xes[i])
            self.assertAlmostEqual(res[0], ys[i][0], places=1)

    def test_gives_correct_output_for_unseen_data(self):
        nnet = NetFactory.create_neural_net(sizes=[1, 10, 1])
        cost_func = QuadraticCost(neural_net=nnet)
        gd = GradientDescent(neural_net=nnet, cost_function=cost_func)

        def parabola(x):
            return x ** 2

        examples = helpers.generate_data(f=parabola, start_value=-0.6,
                                         end_value=-0.4, step_value=0.005)

        gd.train(data_src=PreloadSource(examples), nepochs=10)

        xval = -0.5000125
        yval = parabola(xval)

        net_input = np.array([xval], float)
        output = nnet.feed(net_input)
        self.assertAlmostEqual(output[0], yval, places=1)
