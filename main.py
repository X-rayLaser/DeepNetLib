import numpy as np


def sigma(z):
    return 1.0 / (1.0 + np.exp(z))


class NeuralNet:
    class BadArchitecture(Exception):
        pass

    def __init__(self, layer_sizes):
        if len(layer_sizes) < 3:
            raise self.BadArchitecture('Must be at least 3 layers')

        if layer_sizes[0] <= 0 or layer_sizes[1] <= 0:
            raise self.BadArchitecture('Must have at least 1 node per layer')

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

        z = np.dot(self._weights[layer], activations) + self._biases[layer]
        a = sigma(z)
        return self._feed_next(activations=a, layer=layer+1)

    def feed(self, x):
        return self._feed_next(activations=x, layer=0)

    def train(self, examples):
        pass

    def weights(self):
        return self._weights

    def biases(self):
        return self._biases
