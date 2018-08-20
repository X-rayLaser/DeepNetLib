import unittest
import numpy as np
import helpers
from main import NetFactory
import cost_functions
import activation_functions
from gradient_calculator import BackPropagationBasedCalculator, NumericalCalculator


class ComputeGradientsTests(unittest.TestCase):
    def compare_grads(self, grad1, grad2):
        self.assertTrue(helpers.gradients_equal(grad1, grad2))

    def test_compute_gradients_with_quadratic_cost(self):
        nnet = NetFactory.create_neural_net(sizes=[4, 2, 10])
        nnet.randomize_parameters()
        examples = helpers.generate_random_examples(10, 4, 10)
        calculator = BackPropagationBasedCalculator(examples=examples,
                                                    neural_net=nnet)
        numerical_calculator = NumericalCalculator(examples=examples,
                                                   neural_net=nnet)

        w_grad1, b_grad1 = calculator.compute_gradients()
        w_grad2, b_grad2 = numerical_calculator.compute_gradients()

        self.compare_grads(grad1=w_grad1, grad2=w_grad2)
        self.compare_grads(grad1=b_grad1, grad2=b_grad2)

    def test_compute_gradients_with_cross_entropy_cost(self):
        nnet = NetFactory.create_neural_net(sizes=[4, 2, 10])
        nnet.randomize_parameters()
        nnet.set_cost_function(cost_function=cost_functions.CrossEntropyCost())
        examples = helpers.generate_random_examples(10, 4, 10)
        calculator = BackPropagationBasedCalculator(examples=examples,
                                                    neural_net=nnet)
        numerical_calculator = NumericalCalculator(examples=examples,
                                                   neural_net=nnet)
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
        examples = helpers.generate_random_examples(10, 4, 10)
        calculator = BackPropagationBasedCalculator(examples=examples,
                                                    neural_net=nnet)
        numerical_calculator = NumericalCalculator(examples=examples,
                                                   neural_net=nnet)
        w_grad1, b_grad1 = calculator.compute_gradients()
        w_grad2, b_grad2 = numerical_calculator.compute_gradients()

        self.compare_grads(grad1=w_grad1, grad2=w_grad2)
        self.compare_grads(grad1=b_grad1, grad2=b_grad2)

    def test_that_returned_type_is_array(self):
        nnet = NetFactory.create_neural_net(sizes=[2, 1, 2])
        x = np.array([5, 2], float)
        y = np.array([0.25, 0], float)

        examples = ([x], [y])
        calculator = BackPropagationBasedCalculator(examples=examples,
                                                    neural_net=nnet)

        w_grad, b_grad = calculator.compute_gradients()
        self.assertIsInstance(w_grad, list)
        self.assertIsInstance(w_grad[0], np.ndarray)
        self.assertIsInstance(w_grad[1], np.ndarray)

        self.assertIsInstance(b_grad, list)
        self.assertIsInstance(b_grad[0], np.ndarray)
        self.assertIsInstance(b_grad[1], np.ndarray)

    def test_returns_correct_gradient_shape(self):
        nnet = NetFactory.create_neural_net(sizes=[3, 2, 2, 5])
        x = np.array([5, 2, -0.5], float)
        y = np.array([0.25, 0, 0, 0.7, 0.2], float)
        examples = ([x], [y])
        numerical_calculator = NumericalCalculator(examples=examples,
                                                   neural_net=nnet)
        w_grad, b_grad = numerical_calculator.compute_gradients()
        self.assertEqual(len(w_grad), 3)
        self.assertEqual(len(b_grad), 3)
        self.assertTupleEqual(w_grad[0].shape, (2, 3))
        self.assertTupleEqual(w_grad[1].shape, (2, 2))
        self.assertTupleEqual(w_grad[2].shape, (5, 2))

        self.assertTupleEqual(b_grad[0].shape, (2,))
        self.assertTupleEqual(b_grad[1].shape, (2,))
        self.assertTupleEqual(b_grad[2].shape, (5,))

    def test_with_regularization(self):
        nnet = NetFactory.create_neural_net(sizes=[4, 2, 10])
        nnet.randomize_parameters()
        nnet.set_regularization(reg_lambda=2)
        examples = helpers.generate_random_examples(10, 4, 10)
        calculator = BackPropagationBasedCalculator(examples=examples,
                                                    neural_net=nnet)
        numerical_calculator = NumericalCalculator(examples=examples,
                                                   neural_net=nnet)

        w_grad1, b_grad1 = calculator.compute_gradients()
        w_grad2, b_grad2 = numerical_calculator.compute_gradients()

        self.compare_grads(grad1=w_grad1, grad2=w_grad2)
        self.compare_grads(grad1=b_grad1, grad2=b_grad2)
