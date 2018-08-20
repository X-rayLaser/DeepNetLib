import unittest
import numpy as np
from main import NeuralNet, NetFactory
from activation_functions import Rectifier


class SetActivationFunctionTests(unittest.TestCase):
    def test_feed_outputs_vector_of_zeros(self):
        nnet = NetFactory.create_neural_net(sizes=[2, 4, 3])
        nnet.set_activation_function(activation=Rectifier)
        a = nnet.feed(np.array([2, 10], float))
        self.assertEqual(a.tolist(), [0, 0, 0])

    def test_feed_outputs_correct_vector(self):
        nnet = NetFactory.create_neural_net(sizes=[2, 4, 4])
        nnet.layers()[1].set_biases(np.array([-2, -1, 1, 10], float))

        nnet.set_activation_function(activation=Rectifier)
        a = nnet.feed(np.array([2, 10], float))
        self.assertEqual(a.tolist(), [0, 0, 1, 10])
