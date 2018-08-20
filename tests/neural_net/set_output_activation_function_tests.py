import unittest
import numpy as np
from main import NetFactory
from activation_functions import Sigmoid, Rectifier, Softmax


class SetOutputActivationFunctionTests(unittest.TestCase):
    def test_for_rectifier_feed_outputs_vector_of_zeros(self):
        nnet = NetFactory.create_neural_net(sizes=[2, 4, 3])
        nnet.set_activation_function(activation=Sigmoid)
        nnet.set_output_activation_function(activation=Rectifier)
        a = nnet.feed(np.array([2, 10], float))
        self.assertEqual(a.tolist(), [0, 0, 0])

    def test_for_rectifier(self):
        nnet = NetFactory.create_neural_net(sizes=[2, 4, 4])
        nnet.set_activation_function(activation=Sigmoid)
        nnet.layers()[1].set_biases(np.array([-2, -1, 1, 10], float))

        nnet.set_output_activation_function(activation=Rectifier)
        a = nnet.feed(np.array([2, 10], float))
        self.assertEqual(a.tolist(), [0, 0, 1, 10])

    def test_for_softmax(self):
        nnet = NetFactory.create_neural_net(sizes=[2, 4, 2])
        nnet.set_output_activation_function(activation=Rectifier)

        nnet.layers()[1].set_biases(np.array([3, 3], float))
        nnet.set_output_activation_function(activation=Softmax)
        a = nnet.feed(np.array([2, 10], float))
        self.assertEqual(a.tolist(), [0.5, 0.5])
