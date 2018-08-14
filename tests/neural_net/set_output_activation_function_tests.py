import unittest
import numpy as np
from main import NeuralNet
from activation_functions import Sigmoid, Rectifier, Softmax


class SetOutputActivationFunctionTests(unittest.TestCase):
    def test_for_rectifier_feed_outputs_vector_of_zeros(self):
        nnet = NeuralNet(layer_sizes=[2, 4, 3])
        nnet.set_activation_function(activation=Sigmoid)
        nnet.set_output_activation_function(activation=Rectifier)
        a = nnet.feed(np.array([2, 10], float))
        self.assertEqual(a.tolist(), [0, 0, 0])

    def test_for_rectifier(self):
        nnet = NeuralNet(layer_sizes=[2, 4, 4])
        nnet.set_activation_function(activation=Sigmoid)
        nnet.set_layer_biases(layer=2, bias_vector=np.array([-2, -1, 1, 10], float))

        nnet.set_output_activation_function(activation=Rectifier)
        a = nnet.feed(np.array([2, 10], float))
        self.assertEqual(a.tolist(), [0, 0, 1, 10])

    def test_for_softmax(self):
        nnet = NeuralNet(layer_sizes=[2, 4, 2])
        nnet.set_output_activation_function(activation=Rectifier)

        nnet.set_layer_biases(layer=2, bias_vector=np.array([3, 3], float))
        nnet.set_output_activation_function(activation=Softmax)
        a = nnet.feed(np.array([2, 10], float))
        self.assertEqual(a.tolist(), [0.5, 0.5])