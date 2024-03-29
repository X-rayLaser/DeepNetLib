import numpy as np
import helpers
from neural_net import NetFactory
import cost_functions
import gradient_descent
from data_source import PreloadSource


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
        self._nnet = NetFactory.create_neural_net(sizes=[10, 30, 784])

    def train(self, pixels_to_categories, nepochs=1):
        examples = self.prepare_train_examples(pixels_to_categories)
        cost_func = cost_functions.CrossEntropyCost(self._nnet)
        gd = gradient_descent.StochasticGradientDescent(
            self._nnet, cost_function=cost_func, learning_rate=0.1
        )
        self._nnet.randomize_parameters()
        gd.train(data_src=PreloadSource(examples), nepochs=nepochs)

    def generate(self, seeding_vector):
        x = seeding_vector
        a = self._nnet.feed(x)
        return np.array((a * 255).tolist(), dtype=np.uint8)

    def generate_digit(self, digit):
        if type(digit) != int or digit < 0 or digit > 9:
            raise self.InvalidDigitError(
                'Digit must be an integer from 0 to 9. Got'.format(digit)
            )

        x = helpers.category_to_vector(cat_index=digit, cat_number=10)
        return self.generate(x)

    def save_as_json(self, fname):
        pass

    def load_from_json(self, fname):
        pass

    def net_layer_sizes(self):
        return self._nnet.layer_sizes()
