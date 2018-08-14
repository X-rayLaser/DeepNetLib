import numpy as np
import cost_functions
from gradient_descent import GradientDescent
from activation_functions import sigma, Rectifier, Sigmoid, Softmax


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

    class InvalidMatrixDimensions(Exception):
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
        self._activation_function = Sigmoid
        self._output_activation_function = Sigmoid

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

        if layer == effective_layers_count - 1:
            activ_object = self._output_activation_function
        else:
            activ_object = self._activation_function

        a = activ_object.activation(z)
        return self._feed_next(activations=a, layer=layer+1)

    def feed(self, x):
        if self.x_to_y  and str(x) in self.x_to_y:
            return self.x_to_y[str(x)]

        return self._feed_next(activations=x, layer=0)

    def feed_into_layer(self, x, layer):
        """count starts from first hidden layer"""
        z = weighed_sum(weights=self._weights[layer], activations=x,
                        biases=self._biases[layer])

        effective_layers_count = len(self._sizes) - 1
        if layer == effective_layers_count - 1:
            activ_object = self._output_activation_function
        else:
            activ_object = self._activation_function
        a = activ_object.activation(z)
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

    def set_activation_function(self, activation):
        self._activation_function = activation
        self._output_activation_function = activation

    def set_output_activation_function(self, activation):
        self._output_activation_function = activation

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

    def get_activation_function(self):
        return self._activation_function

    def get_output_activation_function(self):
        return self._output_activation_function

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

    def set_layer_weights(self, layer, weights):
        """layer must be between 1 and number of layers exclusive"""
        if layer < 1 or layer >= self.number_of_layers():
            raise self.LayerOutOfBound(
                'layer must be between 1 and number of layers exclusive'
            )

        if weights.shape != self.weights()[layer - 1].shape:
            raise self.InvalidMatrixDimensions('Wrong weight matrix dimensions')

        self._weights[layer - 1] = np.copy(weights)

    def set_layer_biases(self, layer, bias_vector):
        """layer must be between 1 and number of layers exclusive"""
        if layer < 1 or layer >= self.number_of_layers():
            raise self.LayerOutOfBound(
                'layer must be between 1 and number of layers exclusive'
            )

        if bias_vector.shape != self.biases()[layer - 1].shape:
            raise self.InvalidMatrixDimensions('Wrong weight matrix dimensions')
        self._biases[layer - 1] = np.copy(bias_vector)


    @staticmethod
    def create_from_file(fname):
        import json
        with open(fname, 'r') as f:
            s = f.read()

        net_params = json.loads(s)
        nnet = NeuralNet(layer_sizes=net_params['layer_sizes'])
        for layer in range(1, nnet.number_of_layers()):
            weights = net_params['layers'][layer - 1]['weights']
            biases = net_params['layers'][layer - 1]['biases']
            nnet.set_layer_weights(layer=layer, weights=np.array(weights, float))
            nnet.set_layer_biases(layer=layer, bias_vector=np.array(biases, float))
        return nnet
