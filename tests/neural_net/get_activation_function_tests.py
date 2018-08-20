import unittest
from main import NeuralNet, NetFactory
from activation_functions import Sigmoid, Rectifier


class GetActivationFunctionTests(unittest.TestCase):
    def test_for_default_activation(self):
        nnet = NetFactory.create_neural_net(sizes=[2, 4, 3])
        activation_object = nnet.get_activation_function()

        self.assertEqual(activation_object.__class__, Sigmoid.__class__)

    def test_for_sigmoid(self):
        nnet = NetFactory.create_neural_net(sizes=[2, 4, 3])
        nnet.set_activation_function(Sigmoid)
        activation_object = nnet.get_activation_function()

        self.assertEqual(activation_object, Sigmoid)

    def test_for_rectifier(self):
        nnet = NetFactory.create_neural_net(sizes=[2, 4, 3])
        nnet.set_activation_function(Rectifier)
        activation_object = nnet.get_activation_function()

        self.assertEqual(activation_object, Rectifier)
