import numpy as np
from main import NeuralNet


class DigitGenerator:
    class InvalidDigitError(Exception):
        pass

    @staticmethod
    def prepare_train_examples(pixels_to_categories):
        X, Y = pixels_to_categories
        nexamples = len(Y)
        X_swapped = []
        Y_swapped = []
        for i in range(nexamples):
            X_swapped.append(np.array(Y[i].tolist(), float))
            Y_swapped.append(np.array(X[i].tolist(), float) / 255.0)
        return X_swapped, Y_swapped

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
        return self._nnet.layer_sizes()
