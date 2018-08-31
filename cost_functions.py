"""
A collection of classes encapsulating different loss functions.

For example, in order to compute quadratic cost:
>>> import numpy as np
>>> a = [np.array([2, 4, 3], float), np.array([1, -2, 0], float)]
>>> y = [np.array([2, 4, 3], float), np.array([1, -2, 0], float)]
>>> qc = QuadraticCost()
>>> qc.compute_cost(a, y)
0.0

Classes:
:class CostFunction: a base class providing methods for calculating loss and gradients
:class QuadraticCost: a class represents quadratic loss function
:class CrossEntropyCost: a class represents cross-entropy loss function
:class RegularizedCost: a class for loss function with regularization
"""
import numpy as np
import math


class CostFunction:
    """
    A based class that provides a set of methods to compute loss, partial derivatives
    and other methods useful to implement back propagation algorithm.
    
    Methods:
    :method get_final_layer_error: compute an error in the output layer of neural net
    :method get_error_in_layer: compute an error in some layer
    :method get_weights_gradient: compute a gradient for weights in some layer
    :method get_bias_gradient: compute a gradient for biases in some layer
    :method get_lambda: get the regularization parameter for regularaized loss function 
    """
    """
    This class is supposed to be subclassed. All subclasses must implement methods:
    :method individual_cost: (numpy 1d array, numpy 1d array) => float 
    """

    def get_error_in_layer(self, nabla_next, w_next, z_gradient):
        """
        Calculate an error, nabla, in a layer given the error in the next layer, nabla_next.
        
        :param nabla_next: error in the next layer, numpy 1d array
        :param w_next: matrix of weights in the next layer, numpy 2d array
        :param z_gradient: a vector of partial derivatives of activations, numpy 1d array
        :return: a vector of errors, numpy 1d array
        """
        return w_next.T.dot(nabla_next) * z_gradient

    def get_weights_gradient(self, layer_error, previous_layer_activations):
        """
        Calculate all partial derivatives of loss function with respect to weights in this layer.

        :param layer_error: error nabla in the layer, numpy 1d array 
        :param previous_layer_activations: activations in the previous layer, numpy 1d array
        :return: a matrix of partial derivatives of loss function, numpy 2d array
        """
        nabla = layer_error
        a = previous_layer_activations

        return np.outer(nabla, a)

    def get_final_layer_error(self, activation_last, expected_output, z_gradient_last):
        """
        Calculate a vector of errors in the output layer, nabla.
        
        :param activation_last: actual output vector of neural net, numpy 1d array
        :param expected_output: expected output vector of neural net, numpy 1d array
        :param z_gradient_last: a vector of weighted sums in the output layer, numpy 1d array
        :return: numpy 1d array representing the vector of errors in the output layer, nabla_L
        """
        a_last = activation_last
        y = expected_output
        return (a_last - y) * z_gradient_last

    def get_bias_gradient(self, layer_error):
        """
        Calculate a vector of partial derivatives of loss function with respect to biases in a layer.
        :param layer_error: numpy 1d array of errors in the layer for which you want to find the gradient
        :return: numpy 1d array of partial derivatives of loss function for a given layer
        """
        return layer_error

    def get_lambda(self):
        """
        Get a regularization parameter, lambda.

        :return: a positive floating point number 
        """
        return 0

    def compute_cost(self, activations, outputs):
        """
        Compute a cost.

        :param activations: a list of actual output vectors each of numpy 1d array type, 
        :param outputs: a list of correct output vectors each of numpy 1d array type 
        :return: float
        """
        vector_len = len(activations)

        s = 0
        for i in range(vector_len):
            s += self.individual_cost(activation=activations[i],
                                      expected_output=outputs[i])
        return s / vector_len


class QuadraticCost(CostFunction):
    """
    A subclass of CostFunction that implements an unregularized quadratic loss.
    
    :override compute_cost
    """

    def individual_cost(self, activation, expected_output):
        v = activation - expected_output
        return v.dot(v) / 2.0


class CrossEntropyCost(CostFunction):
    """
        A subclass of CostFunction that implements an unregularized cross-entropy loss.

        :override compute_cost
        :override get_final_layer_error
        """

    def get_final_layer_error(self, activation_last, expected_output, z_gradient):
        a_last = activation_last
        y = expected_output
        return a_last - y

    def _between_numbers(self, a, y):
        """Cross entropy between 2 numbers, a and y defined on [0, 1]"""
        if a == 0 and y == 0:
            return 0

        if a == 1 and y == 1:
            return 0

        if a == 0 or a == 1:
            return np.finfo(float).max

        return -y * math.log(a) - (1 - y) * math.log(1 - a, math.e)

    def individual_cost(self, activation, expected_output):
        a = activation
        y = expected_output
        ncomponents = a.shape[0]
        ce = 0
        for i in range(ncomponents):
            ce += self._between_numbers(a[i], y[i])

        return ce


class RegularizedCost(CostFunction):
    """
    A class-decorator to wrap a loss function object and add L2 regularization.

    :override compute_cost
    :override get_lambda
    """

    def __init__(self, cost_function, regularization_parameter, weights):
        """
        Initialize instance.

        :param cost_function: a subclass of CostFunction class 
        :param regularization_parameter: regularization parameter, lambda, of type float
        :param weights: a python list of 2d numpy arrays - all weights of the neural net
        """
        self._cost_function = cost_function
        self._reglambda = regularization_parameter
        self._weights = weights

    def individual_cost(self, activation, expected_output):
        return self._cost_function.individual_cost(activation, expected_output)

    def compute_cost(self, activations, outputs):
        n = len(outputs)

        square_sum = sum([(w ** 2).sum() for w in self._weights])

        reg_term = self._reglambda / float(2 * n) * square_sum
        old_cost = self._cost_function.compute_cost(activations=activations,
                                                    outputs=outputs)
        return old_cost + reg_term

    def get_lambda(self):
        return self._reglambda


if __name__ == "__main__":
    import doctest

    doctest.testmod()
