import unittest
from main import NeuralNet
from activation_functions import Sigmoid, Rectifier, Softmax


class GetActivationFunctionTests(unittest.TestCase):
    def test_for_default_activation(self):
        nnet = NeuralNet(layer_sizes=[2, 4, 3])
        activation_object = nnet.get_activation_function()

        self.assertEqual(activation_object.__class__, Sigmoid.__class__)

    def test_for_sigmoid(self):
        nnet = NeuralNet(layer_sizes=[2, 4, 3])
        nnet.set_activation_function(Sigmoid)
        activation_object = nnet.get_activation_function()

        self.assertEqual(activation_object, Sigmoid)

    def test_for_rectifier(self):
        nnet = NeuralNet(layer_sizes=[2, 4, 3])
        nnet.set_activation_function(Rectifier)
        activation_object = nnet.get_activation_function()

        self.assertEqual(activation_object, Rectifier)
