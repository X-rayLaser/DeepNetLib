"""
A collection of classes which can be used for calculating gradient of loss function.

Abstract classes:
=======================

    - **NumericalDerivative**: a base abstract class used for calculating numerical derivative
    
    - **GradientCalculator**: a base abstract class used to find gradient


Concrete classes:
=======================

    - **ParameterLocation_**: encapsulates a location of a parameter in a neural net
    
    - **WeightDerivative**: a class used for calculating numerical derivative with respect to weight
    
    - **BiasDerivative**: a class used for calculating numerical derivative with respect to bias
    
    - **NumericalCalculator**: a class which enables one to calculate gradient numerically (inefficient)
    
    - **BackPropagationBasedCalculator**: a class providing efficient way of calculating gradient

API details
=======================
"""
import numpy as np
from backprop import BackPropagation


class GradientCalculator:
    """
    An abstract class
    
    Implement in subclass methods:
        - **compute_gradients**
    """
    def compute_gradients(self):
        """
        Compute gradient of loss function with respect to all weights and biases.
        
        :return: a tuple of 2 python list, items of 1 and 2 lists are, respectively, of type numpy 2d array and numpy 1d array
        """
        raise Exception('Not implemented')


class ParameterLocation:
    def __init__(self, layer, row, column):
        self.layer = layer
        self.row = row
        self.column = column


class NumericalDerivative:
    """
    A base class providing a template for computing numerical derivatives of loss
    function.
    
    Implement in subclass methods:
    - **_increment (parameter_location, epsilon) => None**
    """
    def __init__(self, neural_net, examples, cost_function, epsilon=0.00001):
        """
        Initialize an instance.
        
        :param neural_net: an instance of class 'NeuralNet'
        :param examples: a tuple, (X, Y), where X and Y are lists of numpy 1d arrays 
        :param epsilon: a float representing an interval on which to approximate 
        derivative
        """
        self._neural_net = neural_net
        self._examples = examples
        self._cost_function = cost_function
        self._epsilon = epsilon

    def _derivative(self, cost_minus, cost_plus):
        derivative = (cost_plus - cost_minus) / (2 * self._epsilon)
        return derivative

    def _increment(self, parameter_location, epsilon):
        raise Exception('Not implemented')

    def increment_parameter(self, parameter_location):
        self._increment(parameter_location, self._epsilon)

    def decrement_parameter(self, parameter_location):
        self._increment(parameter_location, -self._epsilon)

    def evaluate_function(self):
        return self._cost_function.get_cost(self._examples)

    def partial_derivative(self, parameter_location):
        """
        Compute a partial derivative of loss function with respect to weight or bias.
        
        :param parameter_location: an instance of ParameterLocation 
        :return: float
        
        **Side effects**: A method temporarily updates a given parameter
        (weight or bias) of neural net.
        """
        self.decrement_parameter(parameter_location)
        cost_minus = self.evaluate_function()

        self.increment_parameter(parameter_location)
        self.increment_parameter(parameter_location)
        cost_plus = self.evaluate_function()

        derivative = self._derivative(cost_minus, cost_plus)

        self.decrement_parameter(parameter_location)
        return derivative


class WeightDerivative(NumericalDerivative):
    """
    A subclass of NumericalDerivative used to compute numerical
    derivatives of loss function with respect to any weight.
    """
    def _increment(self, parameter_location, epsilon):
        neural_net = self._neural_net
        wlist = neural_net.weights()
        weights = wlist[parameter_location.layer]
        current_value = weights[parameter_location.row, parameter_location.column]
        layer = self._neural_net.layers()[parameter_location.layer]
        layer.set_weight(row=parameter_location.row,
                         col=parameter_location.column,
                         new_value=current_value + epsilon)


class BiasDerivative(NumericalDerivative):
    """
    A subclass of NumericalDerivative used to compute numerical
    derivatives of loss function with respect any bias.
    """
    def _increment(self, parameter_location, epsilon):
        neural_net = self._neural_net
        blist = neural_net.biases()
        biases = blist[parameter_location.layer]

        row = parameter_location.row
        layer = self._neural_net.layers()[parameter_location.layer]

        layer.set_bias(row=row, new_value=biases[row] + epsilon)


class NumericalCalculator(GradientCalculator):
    """
    A class to use in order to calculate gradients numerically.
    
    Note that this class should only be used for testing and debugging purposes
    and it is highly inefficient. During production use instance of BackPropagationBasedCalculator
    instead.
    """
    def __init__(self, examples, neural_net, cost_function):
        self._examples = examples
        self._neural_net = neural_net
        self._cost_function = cost_function

    def compute_gradients(self):
        """approximated numerical partial derivatives of loss function"""
        wlist = self._neural_net.weights()
        blist = self._neural_net.biases()

        nmatrices = len(wlist)
        weight_grad = []
        bias_grad = []

        cost_function = self._cost_function
        weight_der = WeightDerivative(neural_net=self._neural_net,
                                      examples=self._examples,
                                      cost_function=cost_function)
        biase_der = BiasDerivative(neural_net=self._neural_net,
                                   examples=self._examples,
                                   cost_function=cost_function)
        for layer in range(nmatrices):
            weight_grad.append(np.zeros(wlist[layer].shape))
            bias_grad.append(np.zeros(blist[layer].shape))

            rows, cols = wlist[layer].shape
            for i in range(rows):
                for j in range(cols):
                    loc = ParameterLocation(layer=layer, row=i, column=j)
                    weight_grad[layer][i][j] = weight_der.partial_derivative(loc)

            for row in range(rows):
                loc = ParameterLocation(layer=layer, row=row, column=0)
                bias_grad[layer][row] = biase_der.partial_derivative(loc)

        return weight_grad, bias_grad


class BackPropagationBasedCalculator(GradientCalculator):
    """
    A class that uses back propagation algorithm to find gradient of loss function.

    During production use instance of this class to calculate gradient as 
    it uses more efficient algorithm than NumericalCalculator.
    """
    def __init__(self, examples, neural_net, cost_function):
        self._examples = examples
        self._neural_net = neural_net
        self._cost_function = cost_function

    def compute_gradients(self):
        xes, ys = self._examples
        examples_count = len(ys)

        weights_grad, biases_grad = self._zero_gradients_list(self._neural_net)

        for i in range(examples_count):
            x = xes[i]
            y = ys[i]
            backprop = BackPropagation(x, y, neural_net=self._neural_net,
                                       cost_function=self._cost_function)
            wgrad, bgrad = backprop.back_propagate()
            weights_grad = self._update_total_gradients(summed_gradients_list=weights_grad,
                                                        new_gradients_list=wgrad)
            biases_grad = self._update_total_gradients(summed_gradients_list=biases_grad,
                                                       new_gradients_list=bgrad)

        weights_grad = self._average_gradient(weights_grad, examples_count)
        biases_grad = self._average_gradient(biases_grad, examples_count)

        return weights_grad, biases_grad

    def _zero_gradients_list(self, neural_net):
        weights_grad = []
        biases_grad = []
        wlist = neural_net.weights()
        blist = neural_net.biases()

        for i in range(len(wlist)):
            wshape = wlist[i].shape
            weights_grad.append(np.zeros(wshape))

            bshape = blist[i].shape
            biases_grad.append(np.zeros(bshape))

        return weights_grad, biases_grad

    def _update_total_gradients(self, summed_gradients_list, new_gradients_list):
        summed_len = len(summed_gradients_list)
        new_len = len(new_gradients_list)
        assert summed_len == new_len

        res_list = []
        for i in range(summed_len):
            res_list.append(summed_gradients_list[i] + new_gradients_list[i])
        return res_list

    def _average_gradient(self, gradient_sum, examples_count):
        res_list = []
        for i in range(len(gradient_sum)):
            res_list.append(gradient_sum[i] / float(examples_count))

        return res_list
