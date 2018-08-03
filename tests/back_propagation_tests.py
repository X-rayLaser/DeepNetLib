import unittest
import numpy as np
import helpers
from backprop import back_propagation
import backprop_slow
from main import NeuralNet


class BackpropagationTests(unittest.TestCase):
    def compare_grads(self, grad1, grad2):
        self.assertTrue(helpers.gradients_equal(grad1, grad2))

    def back_propagation_slow(self, examples, neural_net):
        return backprop_slow.back_propagation_slow(examples=examples, neural_net=neural_net)

    def test_back_propagation_slow(self):
        nnet = NeuralNet(layer_sizes=[1, 1, 1])
        x = np.array([5], float)
        y = np.array([0.25], float)
        examples = ([x], [y])
        w_grad, b_grad = backprop_slow.back_propagation_slow(examples=examples, neural_net=nnet)

        w_grad_expected = [np.array([[0]], float), np.array([[1/32]], float)]
        b_grad_expected = [np.array([[0]], float), np.array([[1/16]], float)]

        self.compare_grads(w_grad, w_grad_expected)
        self.compare_grads(b_grad, b_grad_expected)

    def test_back_propagation_slow_type_array(self):
        nnet = NeuralNet(layer_sizes=[2, 1, 2])
        x = np.array([5, 2], float)
        y = np.array([0.25, 0], float)

        examples = ([x], [y])
        w_grad, b_grad = backprop_slow.back_propagation_slow(examples=examples, neural_net=nnet)
        self.assertIsInstance(w_grad, list)
        self.assertIsInstance(w_grad[0], np.ndarray)
        self.assertIsInstance(w_grad[1], np.ndarray)

        self.assertIsInstance(b_grad, list)
        self.assertIsInstance(b_grad[0], np.ndarray)
        self.assertIsInstance(b_grad[1], np.ndarray)

    def test_back_propagation_slow_shape(self):
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

    def test_back_propagation(self):
        nnet = NeuralNet(layer_sizes=[4, 15, 10])
        examples = helpers.generate_random_examples(10, 4, 10)

        w_grad1, b_grad1 = back_propagation(examples=examples, neural_net=nnet)
        w_grad2, b_grad2 = self.back_propagation_slow(examples=examples, neural_net=nnet)

        self.compare_grads(grad1=w_grad1, grad2=w_grad2)
        self.compare_grads(grad1=b_grad1, grad2=b_grad2)

    def test_back_propagation_type_array(self):
        nnet = NeuralNet(layer_sizes=[2, 1, 2])
        x = np.array([5, 2], float)
        y = np.array([0.25, 0], float)

        examples = ([x], [y])
        w_grad, b_grad = back_propagation(examples=examples, neural_net=nnet)
        self.assertIsInstance(w_grad, list)
        self.assertIsInstance(w_grad[0], np.ndarray)
        self.assertIsInstance(w_grad[1], np.ndarray)

        self.assertIsInstance(b_grad, list)
        self.assertIsInstance(b_grad[0], np.ndarray)
        self.assertIsInstance(b_grad[1], np.ndarray)

    def test_back_propagation_returns_correct_gradient_shape(self):
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