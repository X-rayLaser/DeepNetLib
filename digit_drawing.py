import numpy as np
from main import NeuralNet


class DigitGenerator:
    class InvalidDigitError(Exception):
        pass

    def __init__(self):
        self._nnet = NeuralNet(layer_sizes=[10, 10, 784])

    def train(self):
        pass

    def generate(self, seeding_vector):
        pass

    def generate_digit(self, digit):
        if type(digit) != int or digit < 0 or digit > 9:
            raise self.InvalidDigitError(
                'Digit must be an integer from 0 to 9. Got'.format(digit)
            )
        res = np.zeros(784, dtype=np.uint8)
        res.fill(128)
        return res

    def save_as_json(self, fname):
        pass

    def load_from_json(self, fname):
        pass

    def net_layer_sizes(self):
        sizes = []
        sizes.append(self._nnet.weights()[0].shape[1])
        sizes.append(self._nnet.weights()[0].shape[0])
        sizes.append(self._nnet.weights()[1].shape[0])
        return sizes
