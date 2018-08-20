import unittest
import numpy as np
from main import NeuralNet, NetFactory


class SetBias(unittest.TestCase):
    def test_set_bias(self):
        nnet = NetFactory.create_neural_net(sizes=[2, 3, 2])
        nnet.set_bias(layer=1, row=2, new_value=3.3)
        nnet.set_bias(layer=1, row=1, new_value=-1)
        nnet.set_bias(layer=2, row=0, new_value=2)

        expected_bias1 = np.array([0, -1, 3.3], float)
        expected_bias2 = np.array([2, 0], float)

        b = nnet.biases()
        self.assertTrue(np.allclose(b[0], expected_bias1))
        self.assertTrue(np.allclose(b[1], expected_bias2))

    def test_raises_exception_for_erroneous_layer(self):
        nnet = NetFactory.create_neural_net(sizes=[2, 3, 2, 4])
        self.assertRaises(NeuralNet.LayerOutOfBound,
                          lambda: nnet.set_bias(layer=0, row=1, new_value=2)
                          )

        self.assertRaises(NeuralNet.LayerOutOfBound,
                          lambda: nnet.set_bias(layer=4, row=1, new_value=2)
                          )
        self.assertRaises(NeuralNet.LayerOutOfBound,
                          lambda: nnet.set_bias(layer=14, row=1, new_value=2)
                          )

    def test_raises_exception_for_erroneous_index(self):
        nnet = NetFactory.create_neural_net(sizes=[2, 1, 2])
        self.assertRaises(IndexError,
                          lambda: nnet.set_bias(layer=1, row=1, new_value=2)
                          )
        self.assertRaises(IndexError,
                          lambda: nnet.set_bias(layer=2, row=2, new_value=2)
                          )
