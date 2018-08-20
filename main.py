import numpy as np
import cost_functions
from gradient_descent import GradientDescent
from activation_functions import Sigmoid


def weighed_sum(weights, activations, biases):
    w = weights
    a = activations
    b = biases
    return np.dot(w, a) + b


class CreateLayerMixin:
    def create_next_layer(self, size, activation):
        prev_size = self.get_layer_size()
        return Layer(size, prev_size, activation)


class Layer(CreateLayerMixin):
    class BadArchitecture(Exception):
        pass

    class InvalidMatrixDimensions(Exception):
        pass

    def __init__(self, size, prev_size, activation):
        if size == 0 or prev_size == 0:
            raise self.BadArchitecture('Must have at least 1 node per layer')
        self._weights = np.zeros((size, prev_size), dtype=float)
        self._biases = np.zeros((size, ), dtype=float)
        self._activation_function = activation

    def feed(self, x):
        z = weighed_sum(weights=self.weights(), activations=x,
                        biases=self.biases())
        a = self._activation_function.activation(z)
        return a, z

    def feed_rich(self, x):
        a, z = self.feed(x)
        z_prime = self._activation_function.gradient(z)
        return a, z, z_prime

    def weights(self):
        return self._weights

    def biases(self):
        return self._biases

    def randomize(self):
        rows, cols = self.weights().shape
        self._weights = np.random.randn(rows, cols)

        rows, = self.biases().shape
        self._biases = np.random.randn(rows)

    def set_activation(self, activation):
        self._activation_function = activation

    def get_activation(self):
        return self._activation_function

    def get_layer_size(self):
        return self.biases().shape[0]

    def set_weights(self, weights):
        if weights.shape != self.weights().shape:
            raise self.InvalidMatrixDimensions('Wrong weight matrix dimensions')
        self._weights = np.copy(weights)

    def set_biases(self, biases):
        if biases.shape != self.biases().shape:
            raise self.InvalidMatrixDimensions('Wrong weight matrix dimensions')
        self._biases = np.copy(biases)

    def set_weight(self, row, col, new_value):
        self.weights()[row, col] = new_value

    def set_bias(self, row, new_value):
        self.biases()[row] = new_value


class InputLayer(CreateLayerMixin):
    def __init__(self, size):
        self._size = size

    def get_layer_size(self):
        return self._size


class NetFactory:
    @staticmethod
    def create_neural_net(sizes, hidden_layer_activation=Sigmoid, output_layer_activation=Sigmoid):
        nnet = NeuralNet(layer_sizes=sizes)
        for layer in nnet.layers():
            layer.set_activation(hidden_layer_activation)

        nnet.layers()[-1].set_activation(output_layer_activation)
        return nnet


class NeuralNet:
    class BadArchitecture(Exception):
        pass

    class LayerOutOfBound(Exception):
        pass

    class InvalidMatrixDimensions(Exception):
        pass

    class BadLayerSize(Exception):
        pass

    def __init__(self, layer_sizes):
        if len(layer_sizes) < 3:
            raise self.BadArchitecture('Must be at least 3 layers')

        if layer_sizes[0] <= 0 or layer_sizes[1] <= 0:
            raise self.BadArchitecture('Must have at least 1 node per layer')

        self._sizes = layer_sizes

        self._layers = []

        self._AlgorithmClass = GradientDescent
        self._cost_function = cost_functions.QuadraticCost()
        self._activation_function = Sigmoid
        self._output_activation_function = Sigmoid

        prev_sz = self._sizes[0]
        for sz in self._sizes[1:]:
            self._layers.append(Layer(size=sz, prev_size=prev_sz, activation=Sigmoid))
            prev_sz = sz
        self._set_layers_activations()

    def _set_layers_activations(self):
        for layer in self._layers:
            layer.set_activation(activation=self._activation_function)
        self._layers[-1].set_activation(activation=self._output_activation_function)

    def _feed_next(self, activations, layer):
        effective_layers_count = len(self._sizes) - 1
        if layer >= effective_layers_count:
            return activations

        a, z = self.layers()[layer].feed(activations)
        return self._feed_next(activations=a, layer=layer+1)

    def add_layer(self, layer):
        if layer.weights().shape[1] != self.layers()[-1].weights().shape[0]:
            raise self.BadArchitecture('Incorrect layer')
        self._layers.append(layer)

    def layers(self):
        return self._layers

    def feed(self, x):
        return self._feed_next(activations=x, layer=0)

    def feed_into_layer(self, x, layer):
        """count starts from first hidden layer"""
        return self.layers()[layer].feed(x)

    def train(self, examples, nepochs=1):
        descent = self._AlgorithmClass(neural_net=self)
        descent.train(examples=examples, nepochs=nepochs)

    def weights(self):
        return [layer.weights() for layer in self._layers]

    def biases(self):
        return [layer.biases() for layer in self._layers]

    def number_of_layers(self):
        """Returns total number of layers, including input and output layers"""
        return len(self._layers) + 1

    def layer_sizes(self):
        sizes = []
        weights = self.weights()
        for i in range(len(weights)):
            sizes.append(weights[i].shape[1])

        sizes.append(weights[-1].shape[0])
        return sizes

    def set_cost_function(self, cost_function):
        self._cost_function = cost_function

    def set_regularization(self, reg_lambda=0):
        if reg_lambda > 0 and not isinstance(self._cost_function,
                                             cost_functions.RegularizedCost):
            self._cost_function = cost_functions.RegularizedCost(
                self._cost_function,
                regularization_parameter=reg_lambda,
                weights=self.weights()
            )

    def set_learning_algorithm(self, algorithm_class):
        self._AlgorithmClass = algorithm_class

    def randomize_parameters(self):
        for i in range(len(self.layers())):
            self._layers[i].randomize()

    def get_cost(self, data_set):
        xes, ys = data_set
        activations = [self.feed(x) for x in xes]
        return self._cost_function.compute_cost(activations=activations, outputs=ys)

    def get_cost_function(self):
        return self._cost_function

    def save(self, dest_fname):
        import json
        layers = []

        net_layers = len(self.biases())
        for i in range(net_layers):
            layers.append({
                'weights': self.weights()[i].tolist(),
                'biases': self.biases()[i].tolist()
            })
        net_params = {
            'layer_sizes': self.layer_sizes(),
            'layers': layers
        }
        with open(dest_fname, 'w') as f:
            f.write(json.dumps(net_params))

    @staticmethod
    def create_from_file(fname):
        import json
        with open(fname, 'r') as f:
            s = f.read()

        net_params = json.loads(s)
        nnet = NeuralNet(layer_sizes=net_params['layer_sizes'])
        for layer in range(len(nnet.layers())):
            weights = net_params['layers'][layer]['weights']
            biases = net_params['layers'][layer]['biases']
            nnet.layers()[layer].set_weights(weights=np.array(weights, float))
            nnet.layers()[layer].set_biases(biases=np.array(biases, float))

        return nnet
