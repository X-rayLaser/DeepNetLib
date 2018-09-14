import unittest
import numpy as np
import helpers
from neural_net import NetFactory
import cost_functions
import activation_functions
from gradient_calculator import BackPropagationBasedCalculator, NumericalCalculator
from data_source import PreloadSource


class ComputeGradientsTests(unittest.TestCase):
    def compare_grads(self, grad1, grad2):
        self.assertTrue(helpers.gradients_equal(grad1, grad2))

    def test_compute_gradients_with_quadratic_cost(self):
        nnet = NetFactory.create_neural_net(sizes=[4, 2, 10])
        nnet.randomize_parameters()
        cost_func = cost_functions.QuadraticCost(neural_net=nnet)
        examples = helpers.generate_random_examples(10, 4, 10)
        calculator = BackPropagationBasedCalculator(data_src=PreloadSource(examples),
                                                    neural_net=nnet,
                                                    cost_function=cost_func)
        numerical_calculator = NumericalCalculator(data_src=PreloadSource(examples),
                                                   neural_net=nnet,
                                                   cost_function=cost_func)

        w_grad1, b_grad1 = calculator.compute_gradients()
        w_grad2, b_grad2 = numerical_calculator.compute_gradients()

        self.compare_grads(grad1=w_grad1, grad2=w_grad2)
        self.compare_grads(grad1=b_grad1, grad2=b_grad2)

    def test_compute_gradients_with_cross_entropy_cost(self):
        nnet = NetFactory.create_neural_net(sizes=[4, 2, 10])
        nnet.randomize_parameters()

        cost_func = cost_functions.CrossEntropyCost(neural_net=nnet)
        examples = helpers.generate_random_examples(10, 4, 10)
        calculator = BackPropagationBasedCalculator(data_src=PreloadSource(examples),
                                                    neural_net=nnet,
                                                    cost_function=cost_func)
        numerical_calculator = NumericalCalculator(data_src=PreloadSource(examples),
                                                   neural_net=nnet,
                                                   cost_function=cost_func)
        w_grad1, b_grad1 = calculator.compute_gradients()
        w_grad2, b_grad2 = numerical_calculator.compute_gradients()

        self.compare_grads(grad1=w_grad1, grad2=w_grad2)
        self.compare_grads(grad1=b_grad1, grad2=b_grad2)

    def test_with_rectifer_activation_and_quadratic_cost(self):
        nnet = NetFactory.create_neural_net(
            sizes=[4, 2, 10],
            hidden_layer_activation=activation_functions.Rectifier,
            output_layer_activation=activation_functions.Rectifier
        )
        nnet.randomize_parameters()
        cost_func = cost_functions.QuadraticCost(neural_net=nnet)

        examples = helpers.generate_random_examples(10, 4, 10)
        calculator = BackPropagationBasedCalculator(data_src=PreloadSource(examples),
                                                    neural_net=nnet,
                                                    cost_function=cost_func)
        numerical_calculator = NumericalCalculator(data_src=PreloadSource(examples),
                                                   neural_net=nnet,
                                                   cost_function=cost_func)
        w_grad1, b_grad1 = calculator.compute_gradients()
        w_grad2, b_grad2 = numerical_calculator.compute_gradients()

        self.compare_grads(grad1=w_grad1, grad2=w_grad2)
        self.compare_grads(grad1=b_grad1, grad2=b_grad2)

    def test_that_returned_type_is_array(self):
        nnet = NetFactory.create_neural_net(sizes=[2, 1, 2])
        cost_func = cost_functions.QuadraticCost(neural_net=nnet)

        x = np.array([5, 2], float)
        y = np.array([0.25, 0], float)

        examples = ([x], [y])
        calculator = BackPropagationBasedCalculator(data_src=PreloadSource(examples),
                                                    neural_net=nnet,
                                                    cost_function=cost_func)

        w_grad, b_grad = calculator.compute_gradients()
        self.assertIsInstance(w_grad, list)
        self.assertIsInstance(w_grad[0], np.ndarray)
        self.assertIsInstance(w_grad[1], np.ndarray)

        self.assertIsInstance(b_grad, list)
        self.assertIsInstance(b_grad[0], np.ndarray)
        self.assertIsInstance(b_grad[1], np.ndarray)

    def test_returns_correct_gradient_shape(self):
        nnet = NetFactory.create_neural_net(sizes=[3, 2, 2, 5])
        cost_func = cost_functions.QuadraticCost(neural_net=nnet)

        x = np.array([5, 2, -0.5], float)
        y = np.array([0.25, 0, 0, 0.7, 0.2], float)
        examples = ([x], [y])
        numerical_calculator = NumericalCalculator(data_src=PreloadSource(examples),
                                                   neural_net=nnet,
                                                   cost_function=cost_func)
        w_grad, b_grad = numerical_calculator.compute_gradients()
        self.assertEqual(len(w_grad), 3)
        self.assertEqual(len(b_grad), 3)
        self.assertTupleEqual(w_grad[0].shape, (2, 3))
        self.assertTupleEqual(w_grad[1].shape, (2, 2))
        self.assertTupleEqual(w_grad[2].shape, (5, 2))

        self.assertTupleEqual(b_grad[0].shape, (2,))
        self.assertTupleEqual(b_grad[1].shape, (2,))
        self.assertTupleEqual(b_grad[2].shape, (5,))

    def test_with_regularized_quadratic_loss(self):
        nnet = NetFactory.create_neural_net(sizes=[4, 2, 10])
        nnet.randomize_parameters()

        reglambda = 2.5
        cost_func = cost_functions.QuadraticCost(neural_net=nnet, l2_reg_term=reglambda)
        examples = helpers.generate_random_examples(10, 4, 10)
        calculator = BackPropagationBasedCalculator(data_src=PreloadSource(examples),
                                                    neural_net=nnet,
                                                    cost_function=cost_func)
        numerical_calculator = NumericalCalculator(data_src=PreloadSource(examples),
                                                   neural_net=nnet,
                                                   cost_function=cost_func)

        w_grad1, b_grad1 = calculator.compute_gradients()
        w_grad2, b_grad2 = numerical_calculator.compute_gradients()

        self.compare_grads(grad1=w_grad1, grad2=w_grad2)
        self.compare_grads(grad1=b_grad1, grad2=b_grad2)

    def test_with_regularized_cross_entropy(self):
        nnet = NetFactory.create_neural_net(sizes=[4, 2, 5, 10])
        nnet.randomize_parameters()

        reglambda = 2.5
        cost_func = cost_functions.CrossEntropyCost(neural_net=nnet,
                                                    l2_reg_term=reglambda)
        examples = helpers.generate_random_examples(10, 4, 10)
        calculator = BackPropagationBasedCalculator(data_src=PreloadSource(examples),
                                                    neural_net=nnet,
                                                    cost_function=cost_func)
        numerical_calculator = NumericalCalculator(data_src=PreloadSource(examples),
                                                   neural_net=nnet,
                                                   cost_function=cost_func)

        w_grad1, b_grad1 = calculator.compute_gradients()
        w_grad2, b_grad2 = numerical_calculator.compute_gradients()

        self.compare_grads(grad1=b_grad1, grad2=b_grad2)
        self.compare_grads(grad1=w_grad1, grad2=w_grad2)
