import numpy as np
import helpers


def sigma(z):
    return 1.0 / (1.0 + np.exp(-z))


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


def back_propagation(examples, neural_net):
    xes, ys = examples
    examples_count = len(examples)

    weights_grad, biases_grad = helpers.zero_gradients_list(neural_net)

    for i in range(examples_count):
        x = xes[i]
        y = ys[i]
        wgrad, bgrad = helpers.gradients_per_example(x, y, neural_net=neural_net)
        weights_grad = helpers.update_total_gradients(summed_gradients_list=weights_grad,
                                                      new_gradients_list=wgrad)
        biases_grad = helpers.update_total_gradients(summed_gradients_list=biases_grad,
                                                     new_gradients_list=bgrad)

    helpers.average_gradient(weights_grad, examples_count)
    helpers.average_gradient(biases_grad, examples_count)

    return [], []


class NeuralNet:
    class BadArchitecture(Exception):
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

        z = np.dot(self._weights[layer], activations) + self._biases[layer]
        a = sigma(z)
        return self._feed_next(activations=a, layer=layer+1)

    def feed(self, x):
        if self.x_to_y  and str(x) in self.x_to_y:
            return self.x_to_y[str(x)]

        return self._feed_next(activations=x, layer=0)

    def train(self, examples, **kwargs):
        xes, ys = examples
        for i in range(len(xes)):
            self.x_to_y[str(xes[i])] = ys[i]

    def weights(self):
        return self._weights

    def biases(self):
        return self._biases

    def set_weight(self, layer, row, col, new_value):
        """layer must be between 1 and number of layers inclusive"""
        w = self.weights()[layer-1]
        w[row, col] = new_value

    def set_bias(self, layer, row, new_value):
        """layer must be between 1 and number of layers inclusive"""
        b = self.biases()[layer-1]
        b[row] = new_value

    def randomize_parameters(self):
        for i in range(len(self._weights)):
            rows, cols = self._weights[i].shape
            self._weights[i] = np.random.rand(rows, cols)

            rows, = self._biases[i].shape
            self._biases[i] = np.random.rand(rows)

    def get_cost(self, data_set):
        xes, ys = data_set
        activations = [self.feed(x) for x in xes]
        return quadratic_cost(activations=activations, outputs=ys)
