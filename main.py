import numpy as np
import cost_functions
from gradient_descent import GradientDescent
from activation_functions import sigma


def weighed_sum(weights, activations, biases):
    w = weights
    a = activations
    b = biases
    return np.dot(w, a) + b


class NeuralNet:
    class BadArchitecture(Exception):
        pass

    class LayerOutOfBound(Exception):
        pass

    def __init__(self, layer_sizes):
        if len(layer_sizes) < 3:
            raise self.BadArchitecture('Must be at least 3 layers')

        if layer_sizes[0] <= 0 or layer_sizes[1] <= 0:
            raise self.BadArchitecture('Must have at least 1 node per layer')

        self.x_to_y = {}
        self._sizes = layer_sizes

        self._weights = []
        self._biases = []

        self._AlgorithmClass = GradientDescent
        self._cost_function = cost_functions.QuadraticCost()

        prev_sz = self._sizes[0]
        for sz in self._sizes[1:]:
            shape = (sz, prev_sz)
            w = np.zeros(shape, dtype=float)
            self._weights.append(w)

            b = np.zeros((sz,), dtype=float)
            self._biases.append(b)
            prev_sz = sz

    def _feed_next(self, activations, layer):
        effective_layers_count = len(self._sizes) - 1
        if layer >= effective_layers_count:
            return activations

        z = weighed_sum(weights=self._weights[layer], activations=activations,
                        biases=self._biases[layer])

        a = sigma(z)
        return self._feed_next(activations=a, layer=layer+1)

    def feed(self, x):
        if self.x_to_y  and str(x) in self.x_to_y:
            return self.x_to_y[str(x)]

        return self._feed_next(activations=x, layer=0)

    def feed_into_layer(self, x, layer):
        """count starts from first hidden layer"""
        z = weighed_sum(weights=self._weights[layer], activations=x,
                        biases=self._biases[layer])
        a = sigma(z)
        return a, z

    def train(self, examples, nepochs=1):
        descent = self._AlgorithmClass(neural_net=self)
        descent.train(examples=examples, nepochs=nepochs)

    def weights(self):
        return self._weights

    def biases(self):
        return self._biases

    def number_of_layers(self):
        """Returns total number of layers, including input and output layers"""
        return len(self._biases) + 1

    def layer_sizes(self):
        sizes = []
        weights = self.weights()
        for i in range(len(weights)):
            sizes.append(weights[i].shape[1])

        sizes.append(weights[-1].shape[0])
        return sizes

    def set_weight(self, layer, row, col, new_value):
        """layer must be between 1 and number of layers exclusive"""
        if layer < 1 or layer >= self.number_of_layers():
            raise self.LayerOutOfBound(
                'layer must be between 1 and number of layers exclusive'
            )

        w = self.weights()[layer-1]
        w[row, col] = new_value

    def set_bias(self, layer, row, new_value):
        """layer must be between 1 and number of layers exclusive"""
        if layer < 1 or layer >= self.number_of_layers():
            raise self.LayerOutOfBound(
                'layer must be between 1 and number of layers exclusive'
            )
        b = self.biases()[layer-1]
        b[row] = new_value

    def set_cost_function(self, cost_function):
        self._cost_function = cost_function

    def set_learning_algorithm(self, algorithm_class):
        self._AlgorithmClass = algorithm_class

    def randomize_parameters(self):
        for i in range(len(self._weights)):
            rows, cols = self._weights[i].shape
            self._weights[i] = np.random.randn(rows, cols)

            rows, = self._biases[i].shape
            self._biases[i] = np.random.randn(rows)

    def get_cost(self, data_set):
        xes, ys = data_set
        activations = [self.feed(x) for x in xes]
        return self._cost_function.compute_cost(activations=activations, outputs=ys)

    def get_cost_function(self):
        return self._cost_function
