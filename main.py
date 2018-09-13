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
    """
    A fully-connected layer used by instances of NeuralNet classes.
    
    Can be used as a hidden layer or output layer.
    
    Layer is a subclass of CreateLayerMixin class.
    
    Methods:
        feed: take an input vector and return Z and activation vectors
        feed_rich: similar to feed, but in addition return vector of Z derivatives
        weights: return a matrix of weights
        biases: return a matrix of biases
        randomize: perform initialization of parameters in the layer
        set_activation: set an activation function for this layer
        get_activation: get an activation function
        set_weights: use a given matrix for weights of the layer
        set_biases: use a given vector for biases
        set_weight: set a value for an individual weight
        set_bias: set a values for an bias vector component 
    """
    class BadArchitecture(Exception):
        pass

    class InvalidMatrixDimensions(Exception):
        pass

    def __init__(self, size, prev_size, activation):
        """
        
        :param size: number of units in the layer, int
        :param prev_size: number of units in preceding layer, int
        :param activation: an activation function, one of Sigmoid, Rectifier, Softmax
        """
        if size == 0 or prev_size == 0:
            raise self.BadArchitecture('Must have at least 1 node per layer')
        self._weights = np.zeros((size, prev_size), dtype=float)
        self._biases = np.zeros((size, ), dtype=float)
        self._activation_function = activation

    def feed(self, x):
        """
        For a given vector x, find a weighted sum and activation vectors. 
        :param x: an input vector, a numpy 1d array
        :return: a tuple of activations and weighted sums as numpy 1d arrays
        """
        z = weighed_sum(weights=self.weights(), activations=x,
                        biases=self.biases())
        a = self._activation_function.activation(z)
        return a, z

    def feed_rich(self, x):
        """
        Similarly to feed method, but compute derivatives of weighted sum vector.
        
        :param x: an input vector, a numpy 1d array
        :return: a tuple of activations, weighted sums and their derivatives as numpy 1d arrays 
        """
        a, z = self.feed(x)
        z_prime = self._activation_function.gradient(z)
        return a, z, z_prime

    def weights(self):
        """
        
        :return: return a reference to the weight matrix as numpy 2d array 
        """
        return self._weights

    def biases(self):
        """
        
        :return: return a reference to the bias vector as numpy 1d array 
        """
        return self._biases

    def randomize(self):
        """
        Randomly initialize weights for the purpose of symmetry breaking.
        
        Assign all biases to small positive constant. 
        
        :return: None
        """
        rows, cols = self.weights().shape
        max_val = 0.1
        self._weights = np.random.randn(rows, cols) * max_val

        rows, = self.biases().shape
        mu = 1
        self._biases = np.zeros(rows)
        self._biases.fill(mu)

    def set_activation(self, activation):
        """
        Set an activation function for this layer.
        
        :param activation: one of Sigmoid, Rectifier, Softmax classes 
        :return: None
        """
        self._activation_function = activation

    def get_activation(self):
        """
        Get an activation function used by the layer.
        
        :return: one of Sigmoid, Rectifier, Softmax classes 
        """
        return self._activation_function

    def get_layer_size(self):
        """
        Get a number of units in the layer.
        
        :return: a number of units, int 
        """
        return self.biases().shape[0]

    def set_weights(self, weights):
        """
        Set the exact weights matrix to use.
        
        :param weights: numpy 2d array 
        :return: None
        """
        if weights.shape != self.weights().shape:
            raise self.InvalidMatrixDimensions('Wrong weight matrix dimensions')
        self._weights = np.copy(weights)

    def set_biases(self, biases):
        """
        Set the exact vector of biases.
        
        :param biases: numpy 1d array
        :return: None
        """
        if biases.shape != self.biases().shape:
            raise self.InvalidMatrixDimensions('Wrong weight matrix dimensions')
        self._biases = np.copy(biases)

    def set_weight(self, row, col, new_value):
        """
        Assign a value to the weight in specified location.
        
        :param row: a row in the matrix, int  
        :param col: a column in the matrix, int
        :param new_value: new value of the weight
        :return: None
        """
        self.weights()[row, col] = new_value

    def set_bias(self, row, new_value):
        """
        Assign a value to the bias in specified row.
        
        :param row: a 0-based index of a bias vector 
        :param new_value: new value of the bias
        :return: None
        """
        self.biases()[row] = new_value


class InputLayer(CreateLayerMixin):
    """
    An input layer in the neural net.
    
    No need to include it in a neural net. It serves just
    to simplify the creation of next layers actually present in the
    network structure.
    """
    def __init__(self, size):
        self._size = size

    def get_layer_size(self):
        return self._size


class NetFactory:
    """
    A factory class encapsulating the building an instances of NeuralNet class.
    
    Use this class rather than instantiating NeuralNet objects directly.
    """
    @staticmethod
    def create_neural_net(sizes, hidden_layer_activation=Sigmoid, output_layer_activation=Sigmoid):
        """
        Create a neural net with specified architecture.
        
        :param sizes: a python list containing sizes of all layers except the input layer 
        :param hidden_layer_activation: activation function used in all hidden layers
        :param output_layer_activation: activation function applied in the output layer 
        :return: an instance of NeuralNet class
        """
        input_size = sizes[0]
        nnet = NeuralNet(input_size=input_size)

        layer = InputLayer(size=input_size)
        for size in sizes[1:]:
            layer = layer.create_next_layer(size, activation=hidden_layer_activation)
            nnet.add_layer(layer)

        nnet.layers()[-1].set_activation(output_layer_activation)
        return nnet


class NeuralNet:
    """
    An implementation of an artificial neural network.
    
    Methods:
        add_layer: add an additional layer
        layers: get a list of all layers
        feed: map an input vector x to some output vector y
        feed_into_layer: feed an input directly to the specified layer
        weights: get a list of all weight matrices
        biases: get a list of all biases
        number_of_layers: get a total number of layers
        layer_sizes: get a list of sizes of each layer
        randomize_parameters: randomly initialize weights and biases
        save: save weights and biases in a file
    
    Static methods:
        create_from_file: restore a network from a file
    """
    class BadArchitecture(Exception):
        pass

    def __init__(self, input_size):
        """
        
        :param input_size: number of elements in all input vector 
        """
        if input_size < 1:
            raise self.BadArchitecture('Must have at least 1 node in input layer')

        self._layers = []

    def _feed_next(self, activations, layer):
        if layer >= len(self._layers):
            return activations

        a, z = self.layers()[layer].feed(activations)
        return self._feed_next(activations=a, layer=layer+1)

    def add_layer(self, layer):
        """
        Append a layer to the neural net.
        
        :param layer: an instance of Layer class 
        :return: None
        :raise NeuralNet.BadArchitecture in case there is a mismatch in matrix sizes 
        """
        if len(self.layers()) > 0 and layer.weights().shape[1] != self.layers()[-1].weights().shape[0]:
            raise self.BadArchitecture('Incompatible layers')
        self._layers.append(layer)

    def layers(self):
        """
        
        :return: a python list of Layer instances 
        """
        return self._layers

    def feed(self, x):
        """
        For a given vector x, find an output vector.
        
        :param x: input vector, numpy 1d array 
        :return: output of the network, numpy 1d array
        """
        return self._feed_next(activations=x, layer=0)

    def feed_into_layer(self, x, layer):
        """count starts from first hidden layer"""
        return self.layers()[layer].feed(x)

    def weights(self):
        """
        Get all weights in all layers, 1 matrix per layer.
        
        :return: a python list of weight matrices of type numpy 2d array
        """
        return [layer.weights() for layer in self._layers]

    def biases(self):
        """
        Get all biases in all layers, 1 bias vector per layer.
         
        :return: a python list of bias vectors of type numpy 1d array 
        """
        return [layer.biases() for layer in self._layers]

    def number_of_layers(self):
        """Returns total number of layers, including input and output layers"""
        return len(self._layers) + 1

    def layer_sizes(self):
        """
        Get a list of sizes of all layers except the input layer.
        
        :return: a python list of integers 
        """
        sizes = []
        weights = self.weights()
        for i in range(len(weights)):
            sizes.append(weights[i].shape[1])

        sizes.append(weights[-1].shape[0])
        return sizes

    def randomize_parameters(self):
        for i in range(len(self.layers())):
            self._layers[i].randomize()

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
        layer_sizes = net_params['layer_sizes']
        nnet = NetFactory.create_neural_net(sizes=layer_sizes)
        for layer in range(len(nnet.layers())):
            weights = net_params['layers'][layer]['weights']
            biases = net_params['layers'][layer]['biases']
            nnet.layers()[layer].set_weights(weights=np.array(weights, float))
            nnet.layers()[layer].set_biases(biases=np.array(biases, float))

        return nnet
