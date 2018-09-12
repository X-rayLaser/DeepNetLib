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
from data_source import DataSetIterator


class CostFunction:
    """
    A based class that provides a set of methods to compute loss, partial derivatives
    and other methods useful to implement back propagation algorithm.
    
    Methods:
    :method get_final_layer_error: compute an error in the output layer of neural net
    :method get_lambda: get the regularization parameter for regularaized loss function 
    """
    """
    This class is supposed to be subclassed. All subclasses must implement methods:
    :method individual_cost: (numpy 1d array, numpy 1d array) => float 
    """

    def __init__(self, neural_net, l2_reg_term=0):
        self._net = neural_net
        self._reg_lambda = l2_reg_term

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

    def get_lambda(self):
        """
        Get a regularization parameter, lambda.

        :return: a positive floating point number 
        """
        return self._reg_lambda

    def get_regularized_cost(self, loss):
        square_sum = sum([(w ** 2).sum() for w in self._net.weights()])
        reg_term = 0.5 * self.get_lambda() * square_sum
        return loss + reg_term

    def get_cost(self, data_src):
        """
        Compute a cost.

        :param data_src: instance of DataSource sub class
        :return: float
        """
        it = DataSetIterator(data_src)
        n = data_src.number_of_examples()
        c = 0
        for x, y in it:
            a = self._net.feed(x)
            c += self.individual_cost(activation=a, expected_output=y)

        loss = c / float(n)
        return self.get_regularized_cost(loss)


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


if __name__ == "__main__":
    import doctest

    doctest.testmod()
