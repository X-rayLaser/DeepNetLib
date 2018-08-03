import numpy as np
import helpers
from backprop import back_propagation


def sigma(z):
    return 1.0 / (1.0 + np.exp(-z))


def weighed_sum(weights, activations, biases):
    w = weights
    a = activations
    b = biases
    return np.dot(w, a) + b


def sigma_prime(z):
    return sigma(z) * (1 - sigma(z))


def quadratic_per_example(activation, expected_output):
    v = activation - expected_output
    return v.dot(v) / 2.0


def quadratic_cost(activations, outputs):
    vector_len = len(activations)

    s = 0
    for i in range(vector_len):
        s += quadratic_per_example(activation=activations[i],
                                   expected_output=outputs[i])
    return s / vector_len


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
        descent = GradientDescent(neural_net=self)
        descent.train(examples=examples, nepochs=nepochs)

    def weights(self):
        return self._weights

    def biases(self):
        return self._biases

    def number_of_layers(self):
        """Returns total number of layers, including input and output layers"""
        return len(self._biases) + 1

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

    def randomize_parameters(self):
        for i in range(len(self._weights)):
            rows, cols = self._weights[i].shape
            self._weights[i] = np.random.randn(rows, cols)

            rows, = self._biases[i].shape
            self._biases[i] = np.random.randn(rows)

    def get_cost(self, data_set):
        xes, ys = data_set
        activations = [self.feed(x) for x in xes]
        return quadratic_cost(activations=activations, outputs=ys)


class GradientDescent:
    def __init__(self, neural_net):
        self._nnet= neural_net
        self._rate = 3

    def update_weights(self, weight_gradient):
        weights = self._nnet.weights()
        for i in range(len(weights)):
            weights[i] -= self._rate * weight_gradient[i]

    def update_biases(self, bias_gradient):
        biases = self._nnet.biases()
        for i in range(len(biases)):
            biases[i] -= self._rate * bias_gradient[i]

    def training_epoch(self, examples):
        wgrad, bgrad = back_propagation(examples=examples, neural_net=self._nnet)
        self.update_weights(weight_gradient=wgrad)
        self.update_biases(bias_gradient=bgrad)

    def train(self, examples, nepochs):
        for i in range(nepochs):
            self.training_epoch(examples=examples)
