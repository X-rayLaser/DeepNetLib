import unittest
from neural_net import NeuralNet, NetFactory
import activation_functions


class NetFactoryTests(unittest.TestCase):
    def test_create_neural_net(self):
        nnet = NetFactory.create_neural_net(sizes=[3, 2, 5, 4],
                                            hidden_layer_activation=activation_functions.Rectifier,
                                            output_layer_activation=activation_functions.Softmax)
        self.assertIsInstance(nnet, NeuralNet)
        self.assertEqual(nnet.layer_sizes(), [3, 2, 5, 4])

    def test_create_neural_net_with_rectifier_and_softmax_activations(self):
        nnet = NetFactory.create_neural_net(sizes=[3, 2, 5, 4],
                                            hidden_layer_activation=activation_functions.Rectifier,
                                            output_layer_activation=activation_functions.Softmax)

        for i in range(2):
            layer = nnet.layers()[i]
            self.assertEqual(layer.get_activation(), activation_functions.Rectifier)
        layer = nnet.layers()[-1]
        self.assertEqual(layer.get_activation(), activation_functions.Softmax)

    def test_with_default_activations(self):
        nnet = NetFactory.create_neural_net(sizes=[3, 2, 5, 4])

        for i in range(2):
            layer = nnet.layers()[i]
            self.assertEqual(layer.get_activation(), activation_functions.Sigmoid)
        layer = nnet.layers()[-1]
        self.assertEqual(layer.get_activation(), activation_functions.Sigmoid)
