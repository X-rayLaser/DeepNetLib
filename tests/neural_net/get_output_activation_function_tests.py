import unittest
from main import NeuralNet
from activation_functions import Sigmoid, Rectifier, Softmax


class GetOutputLayerActivationFunctionTests(unittest.TestCase):
    def test_for_sigmoid(self):
        nnet = NeuralNet(layer_sizes=[2, 4, 3])
        from activation_functions import Sigmoid
        nnet.set_output_activation_function(Sigmoid)
        activation_object = nnet.get_output_activation_function()

        self.assertEqual(activation_object, Sigmoid)

    def test_for_rectifier(self):
        nnet = NeuralNet(layer_sizes=[2, 4, 3])
        nnet.set_output_activation_function(Rectifier)
        activation_object = nnet.get_output_activation_function()

        self.assertEqual(activation_object, Rectifier)

    def test_for_softmax(self):
        nnet = NeuralNet(layer_sizes=[2, 4, 3])
        from activation_functions import Softmax
        nnet.set_output_activation_function(Softmax)
        activation_object = nnet.get_output_activation_function()
        self.assertEqual(activation_object, Softmax)
